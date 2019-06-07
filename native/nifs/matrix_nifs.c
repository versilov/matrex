#include <stdint.h>

#include "erl_nif.h"

#include "../include/matrix.h"
#include "../include/matrix_dot.h"
#include "../include/matrix_linalg.h"

#define ASSERT_SIZES_MATCH(m1, m2) if (MX_ROWS(m1) != MX_ROWS(m2) || MX_COLS(m1) != MX_COLS(m2)) \
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

#define UNUSED_VAR(v) (void)(v)

//-----------------------------------------------------------------------------
// Inner helper functions headers
//-----------------------------------------------------------------------------

static double
get_scalar(ErlNifEnv *env, ERL_NIF_TERM arg);

static inline ERL_NIF_TERM
make_cell_value(ErlNifEnv* env, const float value);


//-----------------------------------------------------------------------------
// Exported nifs
//-----------------------------------------------------------------------------

static ERL_NIF_TERM
add(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  float         alpha, beta;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  alpha = get_scalar(env, argv[2]);
  beta = get_scalar(env, argv[3]);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  ASSERT_SIZES_MATCH(first_data, second_data);

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_add(first_data, second_data, alpha, beta, result_data);

  return result;
}

static ERL_NIF_TERM
add_scalar(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  scalar = get_scalar(env, argv[1]);

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
    return enif_raise_exception(env, enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));

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
    return enif_raise_exception(env, enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));

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
concat_columns(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  result_size = 2*sizeof(float) + MX_DATA_BYTE_SIZE(first_data) + MX_DATA_BYTE_SIZE(second_data);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_concat_columns(first_data, second_data, result_data);

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

  ASSERT_SIZES_MATCH(first_data, second_data);

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_divide(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
divide_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  scalar = get_scalar(env, argv[0]);
  if (!enif_inspect_binary(env, argv[1], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_divide_scalar(scalar, matrix_data, result_data);

  return result;
}

static ERL_NIF_TERM
divide_by_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  scalar = get_scalar(env, argv[1]);

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_divide_by_scalar(matrix_data, scalar, result_data);

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

  matrix_dot(1.0, first_data, second_data, result_data);

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

  matrix_dot_and_add(1.0, first_data, second_data, third_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot_and_apply(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  char function_name[16];
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[2], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(env, enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));


  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_COLS(first_data) != MX_ROWS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_ROWS(first_data) * MX_COLS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot_and_apply(1.0, first_data, second_data, function_name, result_data);

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

  matrix_dot_nt(1.0, first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
dot_tn(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  float        *first_data, *second_data, *result_data;
  float       alpha;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  alpha = get_scalar(env, argv[2]);

  first_data  = (float *) first.data;
  second_data = (float *) second.data;

  if (MX_ROWS(first_data) != MX_ROWS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   = MX_COLS(first_data) * MX_COLS(second_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_dot_tn(alpha, first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
forward_substitute(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
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

  if (MX_COLS(first_data) != MX_COLS(first_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));
  if (MX_ROWS(first_data) != MX_ROWS(second_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   =  MX_ROWS(first_data) + 2;

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_solve(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
cholesky(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first;
  ERL_NIF_TERM  result;
  float        *first_data, *result_data;
  int64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);

  first_data  = (float *) first.data;

  if (MX_COLS(first_data) != MX_COLS(first_data))
    return enif_raise_exception(env, enif_make_string(env, "Matrices sizes mismatch.", ERL_NIF_LATIN1));

  data_size   =  MX_ROWS(first_data) * MX_COLS(first_data) + 2;
  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_cholesky(first_data, result_data);

  return result;
}

static ERL_NIF_TERM
eye(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  unsigned long size;
  float value;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_uint64(env, argv[0], &size);
  value = get_scalar(env, argv[1]);


  result_size = (size*size + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, size);
  MX_SET_COLS(result_data, size);
  matrix_eye(result_data, value);

  return result;
}


static ERL_NIF_TERM
diagonal(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  uint32_t      new_rows, new_cols;
  float        *matrix_data, *result_data;
  size_t        result_size;
  uint32_t rows, cols, diag_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);
  diag_size = rows <= cols ? rows : cols;

  new_rows = (uint32_t)1;
  new_cols = (uint32_t)diag_size;

  result_size = sizeof(float) * (2 + new_rows * new_cols);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, new_rows);
  MX_SET_COLS(result_data, new_cols);
  matrix_diagonal(matrix_data, diag_size, result_data);

  return result;
}


static ERL_NIF_TERM
fill(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  ErlNifBinary value;
  unsigned long rows, cols;
  float *value_data;
  float *result_data;
  size_t result_size;

  (void)(argc);

  enif_get_uint64(env, argv[0], &rows);
  enif_get_uint64(env, argv[1], &cols);
  if (!enif_inspect_binary(env, argv[2], &value)) return enif_make_badarg(env);
  value_data = (float*)value.data;

  result_size = (rows*cols + 2) * sizeof(float);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  MX_SET_ROWS(result_data, rows);
  MX_SET_COLS(result_data, cols);

  matrix_fill(result_data, *value_data);

  return result;
}

static ERL_NIF_TERM
find(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, element;
  float        *matrix_data, *element_data;
  int32_t index;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &element)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;
  element_data = (float *) element.data;


  if (isnan(*element_data)) {
    index = matrix_find_nan(matrix_data);
  } else {
    index = matrix_find(matrix_data, *element_data);
  }

  if (index >= 0) {
    int row = index / MX_COLS(matrix_data) + 1;
    int column = index % MX_COLS(matrix_data) + 1;
    return enif_make_tuple2(env, enif_make_int(env, row), enif_make_int(env, column));
  }

  return enif_make_atom(env, "nil");
}

static ERL_NIF_TERM
from_range(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM  result;
  long      from, to, rows, cols;
  float        *result_data;
  size_t        result_size;

  (void)(argc);

  enif_get_int64(env, argv[0], &from);
  enif_get_int64(env, argv[1], &to);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &cols);

  result_size = sizeof(float) * (2 + rows * cols);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_from_range(from, to, rows, cols, result_data);

  return result;
}

// Inner function for getting scalar from args list and casting it to float.
// Scalars come as doubles or long integers.
static double
get_scalar(ErlNifEnv *env, ERL_NIF_TERM arg) {
  double scalar;
  if (enif_get_double(env, arg, &scalar) == 0) {
    long long_scalar;
    enif_get_int64(env, arg, &long_scalar);

    scalar = (double) long_scalar;
  }
  return scalar;
}

static inline ERL_NIF_TERM
make_cell_value(ErlNifEnv* env, const float value) {
  if (isfinite(value))
    return enif_make_double(env, value);
  else if (isnan(value))
    return enif_make_atom(env, "nan");
  else if (value == INFINITY)
    return enif_make_atom(env, "inf");
  else if (value == -INFINITY)
    return enif_make_atom(env, "neg_inf");
  else
    return enif_make_badarg(env);
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

  return make_cell_value(env, max);
}

static ERL_NIF_TERM
max_finite(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float         max;
  float        *matrix_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  max = matrix_max_finite(matrix_data);

  if (isnan(max))
    return enif_make_atom(env, "nil");
  else
    return enif_make_double(env, max);
}

static ERL_NIF_TERM
minimum(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float         min;
  float        *matrix_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  min = matrix_min(matrix_data);

  return make_cell_value(env, min);
}

static ERL_NIF_TERM
min_finite(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float         min;
  float        *matrix_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (float *) matrix.data;

  min = matrix_min_finite(matrix_data);

  if (isnan(min))
    return enif_make_atom(env, "nil");
  else
    return enif_make_double(env, min);
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

  ASSERT_SIZES_MATCH(first_data, second_data);

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
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  scalar = get_scalar(env, argv[1]);

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
normalize(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
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

  matrix_normalize(matrix_data, result_data);

  return result;
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
resize(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  double         scale;
  int32_t      new_rows, new_cols;
  float        *matrix_data, *result_data;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  scale = get_scalar(env, argv[1]);

  matrix_data = (float *) matrix.data;

  new_rows = (int32_t)round((float)MX_ROWS(matrix_data) * scale);
  new_cols = (int32_t)round((float)MX_COLS(matrix_data) * scale);

  result_size = sizeof(float) * (2 + new_rows * new_cols);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_resize(matrix_data, new_rows, new_cols, result_data);

  return result;
}

static ERL_NIF_TERM
row_to_list(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float *matrix_data;
  long row;
  ERL_NIF_TERM  result;
  int32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_int64(env, argv[1], &row);


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
set(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, value;
  float *matrix_data;
  float *result_data;
  uint32_t  result_size;
  unsigned long row;
  unsigned long column;
  float scalar;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &row);
  enif_get_uint64(env, argv[2], &column);
  enif_inspect_binary(env, argv[3], &value);

  scalar = *((float*)value.data);

  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  if (row >= rows || column >= cols)
    return enif_raise_exception(env, enif_make_string(env, "Position out of bounds.", ERL_NIF_LATIN1));


  result_size = MX_BYTE_SIZE(matrix_data);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_set(matrix_data, row, column, scalar, result_data);

  return result;
}

static ERL_NIF_TERM
set_column(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, column_matrix;
  float *matrix_data;
  float *column_matrix_data;
  float *result_data;
  uint32_t  result_size;
  unsigned long column;
  ERL_NIF_TERM  result;
  uint32_t cols;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &column);
  if (!enif_inspect_binary(env, argv[2], &column_matrix)) return enif_make_badarg(env);


  matrix_data = (float *) matrix.data;
  column_matrix_data = (float *) column_matrix.data;
  cols = MX_COLS(matrix_data);

  if (column >= cols)
    return enif_raise_exception(env, enif_make_string(env, "Position out of bounds.", ERL_NIF_LATIN1));


  result_size = MX_BYTE_SIZE(matrix_data);
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_set_column(matrix_data, column, column_matrix_data, result_data);

  return result;
}

static ERL_NIF_TERM
submatrix(ErlNifEnv* env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  float *matrix_data;
  float *result_data;
  uint32_t  result_size;
  unsigned long row_from, row_to, column_from, column_to;
  ERL_NIF_TERM  result;
  uint32_t rows, cols;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &row_from);
  enif_get_uint64(env, argv[2], &row_to);
  enif_get_uint64(env, argv[3], &column_from);
  enif_get_uint64(env, argv[4], &column_to);


  matrix_data = (float *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  if (row_from >= rows || column_from >= cols || row_to >= rows || column_to >= cols)
    return enif_raise_exception(env, enif_make_string(env,
      "Submatrix position out of bounds.", ERL_NIF_LATIN1));


  result_size = ((row_to - row_from + 1) * (column_to - column_from + 1) + 2) * MX_ELEMENT_SIZE;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_submatrix(matrix_data, row_from, row_to, column_from, column_to, result_data);

  return result;
}

static ERL_NIF_TERM
subtract(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
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

  ASSERT_SIZES_MATCH(first_data, second_data);

  data_size   = MX_LENGTH(first_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_subtract(first_data, second_data, result_data);

  return result;
}

static ERL_NIF_TERM
subtract_from_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  float         scalar;
  float        *matrix_data, *result_data;
  uint64_t       data_size;
  size_t        result_size;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[1], &matrix)) return enif_make_badarg(env);
  scalar = get_scalar(env, argv[0]);

  matrix_data = (float *) matrix.data;
  data_size   = MX_LENGTH(matrix_data);

  result_size = sizeof(float) * data_size;
  result_data = (float *) enif_make_new_binary(env, result_size, &result);

  matrix_subtract_from_scalar(scalar, matrix_data, result_data);

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

  return make_cell_value(env, sum);
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
    result = enif_make_list_cell(env, make_cell_value(env, matrix_data[i]), result);
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
      row = enif_make_list_cell(env, make_cell_value(env, matrix_data[2 + cols*r + c]), row);
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
  {"add",                  4, add,                  0},
  {"add_scalar",           2, add_scalar,           0},
  {"apply_math",           2, apply_math,           0},
  {"apply_parallel_math",  2, apply_parallel_math,  0},
  {"argmax",               1, argmax,               0},
  {"column_to_list",       2, column_to_list,       0},
  {"concat_columns",       2, concat_columns,       0},
  {"divide",               2, divide,               0},
  {"divide_scalar",        2, divide_scalar,        0},
  {"divide_by_scalar",     2, divide_by_scalar,     0},
  {"dot",                  2, dot,                  0},
  {"dot_and_add",          3, dot_and_add,          0},
  {"dot_and_apply",        3, dot_and_apply,        0},
  {"dot_nt",               2, dot_nt,               0},
  {"dot_tn",               3, dot_tn,               0},
  {"cholesky",             1, cholesky,             0},
  {"forward_substitute",                2, forward_substitute,                0},
  {"eye",                  2, eye,                  0},
  {"diagonal",             1, diagonal,             0},
  {"fill",                 3, fill,                 0},
  {"find",                 2, find,                 0},
  {"from_range",           4, from_range,           0},
  {"max",                  1, max,                  0},
  {"min",                  1, minimum,              0},
  {"max_finite",           1, max_finite,           0},
  {"min_finite",           1, min_finite,           0},
  {"multiply",             2, multiply,             0},
  {"multiply_with_scalar", 2, multiply_with_scalar, 0},
  {"neg",                  1, neg,                  0},
  {"normalize",            1, normalize,            0},
  {"random",               2, random_matrix,        0},
  {"resize",               2, resize,               0},
  {"row_to_list",          2, row_to_list,          0},
  {"set",                  4, set,                  0},
  {"set_column",           3, set_column,           0},
  {"submatrix",            5, submatrix,            0},
  {"subtract",             2, subtract,             0},
  {"subtract_from_scalar", 2, subtract_from_scalar, 0},
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

// Used for RNG initialization.
int
load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
  // Silence "unused var" warnings.
  (void)(env);
  (void)(priv_data);
  (void)(load_info);

  srandom(time(NULL) + clock());

  return 0;
}

ERL_NIF_INIT(Elixir.Matrex.NIFs, nif_functions, load, NULL, upgrade, NULL)
