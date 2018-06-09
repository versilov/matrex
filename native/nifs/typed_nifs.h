TYPED_NIF(add_arrays, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  TYPE        *first_data, *second_data, *result_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size/sizeof(TYPE); i++)
    result_data[i] = first_data[i] + second_data[i];

  return result;
}

TYPED_NIF(add_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  ERL_NIF_TERM  result;
  TYPE        *array_data, *result_data;
  TOP_TYPE scalar;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &array)) return enif_make_badarg(env);
  ENIF_GET_VAL(scalar, argv[1]);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size/sizeof(TYPE); i++)
    result_data[i] = array_data[i] + scalar;

  return result;
}

#ifdef BLAS_GEMM

TYPED_NIF(dot_arrays, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *result_data;
  long rows, dim, cols;
  double alpha;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &dim);
  enif_get_int64(env, argv[4], &cols);
  get_scalar_float(env, argv[5], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  BLAS_GEMM(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    rows,
    cols,
    dim,
    alpha,
    first_data,
    dim,
    second_data,
    cols,
    0.0,
    result_data,
    cols
  );

  return result;
}
#else

// Implement naive integer matrix dot

#endif

TYPED_NIF(multiply_arrays, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  TYPE        *first_data, *second_data, *result_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size/sizeof(TYPE); i++)
    result_data[i] = first_data[i] * second_data[i];

  return result;
}

TYPED_NIF(ones_array, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  TYPE *result_data;
  long size;

  (void)(argc);

  enif_get_int64(env, argv[0], &size);

  result_data = (TYPE*)enif_make_new_binary(env, size * sizeof(TYPE), &result);

  for (uint64_t i = 0; i < size; i++)
    result_data[i] = (TYPE)1;

  return result;
}

TYPED_NIF(random_array, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  TYPE *result_data;
  unsigned long size;

  (void)(argc);

  enif_get_int64(env, argv[0], &size);

  result_data = (TYPE*)enif_make_new_binary(env, size * sizeof(TYPE), &result);

  for (uint64_t i = 0; i < size; i++)
    result_data[i] = (TYPE)random()/(TYPE)RAND_MAX;

  return result;
}

TYPED_NIF(array_sum, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  TYPE        *array_data;
  TOP_TYPE sum = 0;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &array)) return enif_make_badarg(env);

  array_data  = (TYPE*)array.data;

  for (uint64_t i = 0; i < array.size/sizeof(TYPE); i++)
    sum += array_data[i];

  return ENIF_MAKE_VAL(sum);
}

// NIFs defined only for floating point types (float32 & float64)
#ifdef FLOAT_NIFS

#define MATH_FUNC_TYPE(TN) CAT(math_func_, TN, _ptr_t)
#define MATH_FUNC_FROM_NAME(TN) CAT(math_func_, TN, _from_name)

TYPED_NIF(array_apply_math, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  char          function_name[16];
  ERL_NIF_TERM  result;
  TYPE *array_data, *result_data;

  (void)(argc);

  if (!enif_inspect_binary(env, argv[0], &array )) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[1], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(
      env, 
      enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));

  MATH_FUNC_TYPE(TYPE_NAME) func = MATH_FUNC_FROM_NAME(TYPE_NAME)(function_name);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size/sizeof(TYPE); i++)
    result_data[i] = func(array_data[i]);

  return result;
}

#endif
