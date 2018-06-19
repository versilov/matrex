#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

#include "erl_nif.h"
#include "../include/matrix.h"

typedef unsigned char byte;
typedef int64_t int64;
typedef uint64_t uint64;


#define CAT(a, b, c) a ## b ## c

#define TYPED_NIF(title, type_name) static ERL_NIF_TERM CAT(title, _, type_name)

#define ENIF_MAKE_VAL(VAL, TYPE) CAT(make, _, TYPE)(env, VAL)

#define ENIF_GET_VAL(VAL, ARG, TYPE) CAT(get_scalar, _, TYPE)(env, ARG, &VAL)

#define UNUSED_VAR(v) (void)(v)

static void
get_scalar_double(ErlNifEnv *env, ERL_NIF_TERM arg, double* scalar) {
  if (enif_get_double(env, arg, scalar) == 0) {
    long long_scalar;
    enif_get_int64(env, arg, &long_scalar);

    *scalar = (double) long_scalar;
  }
}

static void
get_scalar_int64(ErlNifEnv *env, ERL_NIF_TERM arg, int64_t* scalar) {
  if (enif_get_int64(env, arg, (long*)scalar) == 0) {
    double double_scalar;
    enif_get_double(env, arg, &double_scalar);

    *scalar = (long) double_scalar;
  }
}

static inline ERL_NIF_TERM
make_int64(ErlNifEnv* env, const int64_t value) {
  return enif_make_int64(env, value);
}


static inline ERL_NIF_TERM
make_double(ErlNifEnv* env, const double value) {
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
zeros(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  long byte_size;
  void *result_data;

  UNUSED_VAR(argc);

  enif_get_int64(env, argv[0], &byte_size);

  result_data = enif_make_new_binary(env, byte_size, &result);
  memset(result_data, 0, byte_size);

  return result;
}

#define TOP_TYPE int64

#define TYPE uint8_t
#define TYPE_NAME byte
#include "nifs.h"

#undef TYPE
#undef TYPE_NAME
#define TYPE int16_t
#define TYPE_NAME int16
#include "nifs.h"

#undef TYPE
#undef TYPE_NAME
#define TYPE int32_t
#define TYPE_NAME int32
#include "nifs.h"

#undef TYPE
#undef TYPE_NAME
#define TYPE int64_t
#define TYPE_NAME int64
#include "nifs.h"

#undef TOP_TYPE
#define TOP_TYPE double
#define FLOAT_NIFS 1

#undef TYPE
#undef TYPE_NAME
#define BLAS_GEMM cblas_sgemm
#define TYPE float
#define TYPE_NAME float32
#include "nifs.h"

#undef TYPE
#undef TYPE_NAME
#undef BLAS_GEMM
#define BLAS_GEMM cblas_dgemm
#define TYPE double
#define TYPE_NAME float64
#include "nifs.h"

#undef FLOAT

#define TYPED_NIFS_DECL(NAME, ARGC, FLAGS) \
  {#NAME "_byte", ARGC, NAME##_byte, FLAGS}, \
  {#NAME "_int16", ARGC, NAME##_int16, FLAGS}, \
  {#NAME "_int32", ARGC, NAME##_int32, FLAGS}, \
  {#NAME "_int64", ARGC, NAME##_int64, FLAGS}, \
  {#NAME "_float32", ARGC, NAME##_float32, FLAGS}, \
  {#NAME "_float64", ARGC, NAME##_float64, FLAGS}

#define FLOAT_TYPED_NIFS_DECL(NAME, ARGC, FLAGS) \
  {#NAME "_float32", ARGC, NAME##_float32, FLAGS}, \
  {#NAME "_float64", ARGC, NAME##_float64, FLAGS}

#define NIF_DECL(NAME, ARGC, FLAGS) {#NAME, ARGC, NAME, FLAGS}

static ErlNifFunc nif_functions[] = {
  TYPED_NIFS_DECL(add_scalar, 3, 0),
  TYPED_NIFS_DECL(add, 4, 0),
  FLOAT_TYPED_NIFS_DECL(apply_math, 2, 0),
  TYPED_NIFS_DECL(argmax, 1, 0),
  TYPED_NIFS_DECL(column_to_list, 3, 0),
  TYPED_NIFS_DECL(concat_columns, 4, 0),
  TYPED_NIFS_DECL(divide_by_scalar, 2, 0),
  TYPED_NIFS_DECL(divide_scalar, 2, 0),
  TYPED_NIFS_DECL(divide, 3, 0),
  TYPED_NIFS_DECL(dot, 6, 0),
  TYPED_NIFS_DECL(dot_and_add, 7, 0),
  TYPED_NIFS_DECL(dot_and_appply, 7, 0),
  TYPED_NIFS_DECL(dot_nt, 6, 0),
  TYPED_NIFS_DECL(dot_tn, 6, 0),
  TYPED_NIFS_DECL(eye, 2, 0),
  TYPED_NIFS_DECL(fill, 2, 0),
  TYPED_NIFS_DECL(find, 2, 0),
  TYPED_NIFS_DECL(max, 1, 0),
  FLOAT_TYPED_NIFS_DECL(max_finite, 1, 0),
  TYPED_NIFS_DECL(min, 1, 0),
  FLOAT_TYPED_NIFS_DECL(min_finite, 1, 0),
  TYPED_NIFS_DECL(multiply, 3, 0),
  TYPED_NIFS_DECL(multiply_with_scalar, 2, 0),
  TYPED_NIFS_DECL(neg, 1, 0),
  FLOAT_TYPED_NIFS_DECL(normalize, 1, 0),
  TYPED_NIFS_DECL(random, 1, 0),
  TYPED_NIFS_DECL(resize, 4, 0),
  TYPED_NIFS_DECL(from_range, 2, 0),
  TYPED_NIFS_DECL(row_to_list, 2, 0),
  TYPED_NIFS_DECL(set, 3, 0),
  TYPED_NIFS_DECL(set_column, 3, 0),
  TYPED_NIFS_DECL(submatrix, 5, 0),
  TYPED_NIFS_DECL(subtract, 4, 0),
  TYPED_NIFS_DECL(subtract_from_scalar, 3, 0),
  TYPED_NIFS_DECL(sum, 1, 0),
  TYPED_NIFS_DECL(to_list, 1, 0),
  TYPED_NIFS_DECL(to_list_of_lists, 3, 0),
  TYPED_NIFS_DECL(transpose, 3, 0),
  NIF_DECL(zeros, 1, 0)
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
