#include <stdint.h>
#include <string.h>
#include <time.h>

#include "erl_nif.h"

typedef unsigned char byte;

void
_add_arrays(const void* first, const void* second, void* result, uint64_t byte_size, char* data_type);


static ERL_NIF_TERM
add_arrays(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  void        *first_data, *second_data, *result_data;
  char data_type[16];

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[2], data_type, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(env,
      enif_make_string(env, "Third argument must be an atom.", ERL_NIF_LATIN1));


  first_data  = first.data;
  second_data = second.data;

  result_data = enif_make_new_binary(env, first.size, &result);

  _add_arrays(first_data, second_data, result_data, first.size, data_type);

  return result;
}

void
_add_arrays(const void* first, const void* second, void* result, uint64_t byte_size, char* data_type) {
  if (strcmp(data_type, "float32") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(float); i++)
      ((float*)result)[i] = ((float*)first)[i] + ((float*)second)[i];
  } else if (strcmp(data_type, "byte") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(byte); i++)
      ((byte*)result)[i] = ((byte*)first)[i] + ((byte*)second)[i];
  } else if (strcmp(data_type, "float64") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(double); i++)
      ((double*)result)[i] = ((double*)first)[i] + ((double*)second)[i];
  } else if (strcmp(data_type, "int16") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(int16_t); i++)
      ((int16_t*)result)[i] = ((int16_t*)first)[i] + ((int16_t*)second)[i];
  } else if (strcmp(data_type, "int32") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(int32_t); i++)
      ((int32_t*)result)[i] = ((int32_t*)first)[i] + ((int32_t*)second)[i];
  } else if (strcmp(data_type, "int64") == 0) {
    for (uint64_t i = 0; i < byte_size/sizeof(int64_t); i++)
      ((int64_t*)result)[i] = ((int64_t*)first)[i] + ((int64_t*)second)[i];
  }
}


static ErlNifFunc nif_functions[] = {
  {"add_arrays",                  3, add_arrays,                  0}
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

ERL_NIF_INIT(Elixir.Matrex.Array.NIFs, nif_functions, load, NULL, upgrade, NULL)
