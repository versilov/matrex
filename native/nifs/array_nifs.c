#include <stdint.h>
#include <string.h>
#include <time.h>

#include "erl_nif.h"

typedef unsigned char byte;

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

  if (strcmp(data_type, "float32") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(float); i++)
      ((float*)result_data)[i] = ((float*)first_data)[i] + ((float*)second_data)[i];
  } else if (strcmp(data_type, "byte") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(byte); i++)
      ((byte*)result_data)[i] = ((byte*)first_data)[i] + ((byte*)second_data)[i];
  } else if (strcmp(data_type, "float64") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(double); i++)
      ((double*)result_data)[i] = ((double*)first_data)[i] + ((double*)second_data)[i];
  } else if (strcmp(data_type, "int16") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(int16_t); i++)
      ((int16_t*)result_data)[i] = ((int16_t*)first_data)[i] + ((int16_t*)second_data)[i];
  } else if (strcmp(data_type, "int32") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(int32_t); i++)
      ((int32_t*)result_data)[i] = ((int32_t*)first_data)[i] + ((int32_t*)second_data)[i];
  } else if (strcmp(data_type, "int64") == 0) {
    for (uint64_t i = 0; i < first.size/sizeof(int64_t); i++)
      ((int64_t*)result_data)[i] = ((int64_t*)first_data)[i] + ((int64_t*)second_data)[i];
  }

  return result;
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
