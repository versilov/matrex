#include <stdint.h>

#include "erl_nif.h"

#include "../include/matrix.h"

//-----------------------------------------------------------------------------
// Exported nifs
//-----------------------------------------------------------------------------

static ERL_NIF_TERM
add(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data) || MX_COLS(first_data) != MX_COLS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_add(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
add_scalar(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  double        large_scalar;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  if (enif_get_double(env, argv[1], &large_scalar) == 0) {
    long long_element;
    enif_get_int64(env, argv[1], &long_element);

    large_scalar = (double) long_element;
  }
  scalar = (float) large_scalar;

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_add_scalar(matrix_data, scalar, result_data);

  return result;
}

static ERL_NIF_TERM
apply_math(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  char          function_name[16];
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[1], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(env, enif_make_atom(env, "second_argument_must_be_an_atom"));

  matrix_data  = (float *) matrix.data;

  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  if (matrix_apply(matrix_data, function_name, result_data) == 1)
    return result;
  else
    return enif_make_badarg(env);
}

#define WORKERS_NUM 8

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void* math(void* args) {
  const Matrix* m = (const Matrix*)args;
  const Matrix matrix = m[0];
  const Matrix result = m[1];
  const uint64_t* i= (const uint64_t*)args;
  const uint64_t from = i[2];
  const uint64_t to = i[3];
  math_func_ptr_t* f = (math_func_ptr_t*)args;
  math_func_ptr_t math_func = f[4];

  for (uint64_t index = 2 + from; index < (2 + to); index += 1) {
    result[index] = math_func(matrix[index]);
  }

  return NULL;
}

static ERL_NIF_TERM
apply_parallel_math(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  char          function_name[16];
  float        *matrix_data, *result_data;
  uint64_t       data_size, chunk_size;
  size_t        result_size;
  ErlNifTid workers[WORKERS_NUM];

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[1], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(env, enif_make_atom(env, "second_argument_must_be_an_atom"));

  matrix_data  = (float *) matrix.data;

  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);


  MX_SET_ROWS(result_data, MX_ROWS(matrix_data));
  MX_SET_COLS(result_data, MX_COLS(matrix_data));

  chunk_size = data_size / WORKERS_NUM + 1;

  math_func_ptr_t func = math_func_from_name(function_name);

  for (int i = 0; i < WORKERS_NUM; i++ ) {
    uint64_t from = i*chunk_size;
    uint64_t to = min((i+1)*chunk_size, data_size-2);
    void* args[] = {matrix_data, result_data, (void*)from, (void*)to, (void*)func};
    enif_thread_create("apply_math", &workers[i], &math, args, NULL);
  }

  for (int i = 0; i < WORKERS_NUM; i++ ) {
    enif_thread_join(workers[i], NULL);
  }

  return result;
}

static ERL_NIF_TERM
argmax(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float        *matrix_data;
  int32_t       argmax;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  argmax      = matrix_argmax(matrix_data);

  return enif_make_int(env, argmax);
}

static ERL_NIF_TERM
column_to_list(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float *matrix_data;
  long column;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_int64(env, argv[1], &column);


  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  if (column >= cols)
    return enif_raise_exception(env, enif_make_string(env, "Column index out of bounds.", ERL_NIF_LATIN1));

  result = enif_make_list(env, 0);

  for (int64_t i = (rows-1)*cols + column + 2; i >= (column + 2); i -= cols ) {
    result = enif_make_list_cell(env, enif_make_double(env, matrix_data[i]), result);
  }

  return result;
}

static ERL_NIF_TERM
divide(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  int32_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data) || MX_COLS(first_data) != MX_COLS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_divide(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  int64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_COLS(first_data) != MX_ROWS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   =  MX_ROWS(first_data) * MX_COLS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot_and_add(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second, third;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *third_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[2], &third )) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;
  third_data  = (float *) third.data;

  if (MX_COLS(first_data) != MX_ROWS(second_data) ||
      MX_ROWS(first_data) != MX_ROWS(third_data) ||
      MX_COLS(second_data) != MX_COLS(third_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_ROWS(first_data) * MX_COLS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot_and_add(first_data, second_data, third_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot_nt(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_COLS(first_data) != MX_COLS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_ROWS(first_data) * MX_ROWS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot_nt(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot_tn(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_COLS(first_data) * MX_COLS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot_tn(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
eye(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  unsigned long size;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_uint64(env, argv[0], &size);

  result_size = (size*size + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, size);
  MX_SET_COLS(result_data, size);
  matrix_eye(result_data, 1);

  return result;
}

static ERL_NIF_TERM
fill(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  unsigned long rows, cols, value;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_uint64(env, argv[0], &rows);
  enif_get_uint64(env, argv[1], &cols);
  enif_get_uint64(env, argv[2], &value);

  result_size = (rows*cols + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, rows);
  MX_SET_COLS(result_data, cols);

  matrix_fill(result_data, value);

  return result;
}

static ERL_NIF_TERM
max(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float         max;
  float        *matrix_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  max = matrix_max(matrix_data);

  return enif_make_double(env, max);
}

static ERL_NIF_TERM
multiply(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data) || MX_COLS(first_data) != MX_COLS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_multiply(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
multiply_with_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  double        large_scalar;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  if (enif_get_double(env, argv[1], &large_scalar) == 0) {
    long long_element;
    enif_get_int64(env, argv[1], &long_element);

    large_scalar = (double) long_element;
  }
  scalar = (float) large_scalar;

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_multiply_with_scalar(matrix_data, scalar, result_data);

  return result;
}

static ERL_NIF_TERM
neg(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_neg(matrix_data, result_data);

  return result;
}



static ERL_NIF_TERM
wrap_matrex(ErlNifEnv *env, ERL_NIF_TERM matrix_binary) {
  // we make an empty erlang/elixir map object
  ERL_NIF_TERM matrex = enif_make_new_map(env);

  enif_make_map_put(env, matrex,
                      enif_make_atom(env, "__struct__"),
                      enif_make_atom(env, "Elixir.Matrex"),
                      &matrex);

  enif_make_map_put(env, matrex,
                      enif_make_atom(env, "data"),
                      matrix_binary,
                      &matrex);

  return matrex;
}

static ERL_NIF_TERM
random_matrix(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  long rows, cols;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_int64(env, argv[0], &rows);
  enif_get_int64(env, argv[1], &cols);

  result_size = (rows*cols + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, rows);
  MX_SET_COLS(result_data, cols);

  matrix_random(result_data);

  return result;
}

static ERL_NIF_TERM
row_to_list(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float *matrix_data;
  unsigned long row;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &row);


  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  if (row >= rows)
    return enif_raise_exception(env, enif_make_string(env, "Row index out of bounds.", ERL_NIF_LATIN1));

  result = enif_make_list(env, 0);

  for (int64_t i = (row+1)*cols + 2; i-- > row*cols + 2; ) {
    result = enif_make_list_cell(env, enif_make_double(env, matrix_data[i]), result);
  }

  return result;
}


static ERL_NIF_TERM
substract(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  int32_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data) || MX_COLS(first_data) != MX_COLS(second_data))
      return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_substract(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
substract_from_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  double        large_scalar;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[1], &matrix)) return enif_make_badarg(env);
  if (enif_get_double(env, argv[0], &large_scalar) == 0) {
    long long_element;
    enif_get_int64(env, argv[0], &long_element);

    large_scalar = (double) long_element;
  }
  scalar = (float) large_scalar;

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_substract_from_scalar(scalar, matrix_data, result_data);

  return result;
}

static ERL_NIF_TERM
sum(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  double         sum;
  float        *matrix_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  sum = matrix_sum(matrix_data);

  return enif_make_double(env, sum);
}

static ERL_NIF_TERM
to_list(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  /* to_list(matrix) -> [first row, second row, ...,last row] */
  ErlNifBinary  matrix;
  float *matrix_data;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  result = enif_make_list(env, 0);

  for (uint64_t i = rows*cols + 2; i-- > 2; ) {
    result = enif_make_list_cell(env, enif_make_double(env, matrix_data[i]), result);
  }

  return result;
}

static ERL_NIF_TERM
to_list_of_lists(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  /* to_list_of_lists(matrix) -> [[first row], [second row], ...,[last row]] */
  ErlNifBinary  matrix;
  float *matrix_data;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  result = enif_make_list(env, 0);

  for (uint32_t r = rows; r-- > 0; ) {
    ERL_NIF_TERM row = enif_make_list(env, 0);
    for (uint32_t c = cols; c-- > 0; ) {
      row = enif_make_list_cell(env, enif_make_double(env, matrix_data[2 + cols*r + c]), row);
    }
    result = enif_make_list_cell(env, row, result);
  }

  return result;
}

static ERL_NIF_TERM
transpose(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_transpose(matrix_data, result_data);

  return result;
}

static ERL_NIF_TERM
zeros(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  long rows, cols;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_int64(env, argv[0], &rows);
  enif_get_int64(env, argv[1], &cols);

  result_size = (rows*cols + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, rows);
  MX_SET_COLS(result_data, cols);

  matrix_zeros(result_data);

  return result;
}

static ErlNifFunc nif_functions[] = {
  {"add",                  2, add,                  0},
  {"add_scalar",           2, add_scalar,           0},
  {"apply_math",           2, apply_math,           0},
  {"apply_parallel_math",  2, apply_parallel_math,  0},
  {"argmax",               1, argmax,               0},
  {"column_to_list",       2, column_to_list,       0},
  {"divide",               2, divide,               0},
  {"dot",                  2, dot,                  0},
  {"dot_and_add",          3, dot_and_add,          0},
  {"dot_nt",               2, dot_nt,               0},
  {"dot_tn",               2, dot_tn,               0},
  {"eye",                  1, eye,                  0},
  {"fill",                 3, fill,                 0},
  {"max",                  1, max,                  0},
  {"multiply",             2, multiply,             0},
  {"multiply_with_scalar", 2, multiply_with_scalar, 0},
  {"neg",                  1, neg,                  0},
  {"random",               2, random_matrix,        0},
  {"row_to_list",          2, row_to_list,          0},
  {"substract",            2, substract,            0},
  {"substract_from_scalar",2, substract_from_scalar,0},
  {"sum",                  1, sum,                  0},
  {"to_list",              1, to_list,              0},
  {"to_list_of_lists",     1, to_list_of_lists,     0},
  {"transpose",            1, transpose,            0},
  {"zeros",                2, zeros,                0}
};

// Solely to silence coveralls.travis task errors on Travis CI
int
upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data, ERL_NIF_TERM load_info) {
  // Silence "unused var" warnings.
  (void)(env);
  (void)(priv_data);
  (void)(old_priv_data);
  (void)(load_info);
  return 0;
}

ERL_NIF_INIT(Elixir.Matrex.NIFs, nif_functions, NULL, NULL, upgrade, NULL)
