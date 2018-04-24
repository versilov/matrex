#include <stdint.h>

#include "erl_nif.h"

#include "../include/matrix.h"

#define WORKERS_NUM 8

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef float (*math_func_ptr_t)(float);

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
apply_math(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
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

static ErlNifFunc nif_functions[] = {
  {"apply_math",            2, apply_math,           0}
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

ERL_NIF_INIT(Elixir.Matrex.Threaded, nif_functions, NULL, NULL, upgrade, NULL)
