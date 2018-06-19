TYPED_NIF(add, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  TYPE        *first_data, *second_data, *result_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size / sizeof(TYPE); i++)
    result_data[i] = first_data[i] + second_data[i];

  return result;
}

TYPED_NIF(add_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  ERL_NIF_TERM  result;
  TYPE        *array_data, *result_data;
  TOP_TYPE scalar;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &array)) return enif_make_badarg(env);
  ENIF_GET_VAL(scalar, argv[1], TOP_TYPE);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size / sizeof(TYPE); i++)
    result_data[i] = array_data[i] + scalar;

  return result;
}


TYPED_NIF(argmax, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(column_to_list, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(concat_columns, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *result_data;
  long columns1, columns2, rows, result_cols;
  size_t result_size;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);

  enif_get_int64(env, argv[2], &columns1);
  enif_get_int64(env, argv[3], &columns2);

  first_data = (TYPE*)first.data;
  second_data = (TYPE*)second.data;
  result_size = first.size + second.size;
  result_cols = columns1 + columns2;
  rows = first.size / (columns1 * sizeof(TYPE));

  result_data = (TYPE*) enif_make_new_binary(env, result_size, &result);

  for (int64_t row = 0; row < rows; row++) {
    memcpy(&result_data[row*result_cols], &first_data[row*columns1], columns1*sizeof(TYPE));
    memcpy(&result_data[row*result_cols + columns1], &second_data[row*columns2], columns2*sizeof(TYPE));
  }

  return result;
}

TYPED_NIF(divide_by_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(divide_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(divide, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  ErlNifBinary  first, second;
  TYPE *first_data, *second_data, *result_data;
  TOP_TYPE alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  ENIF_GET_VAL(alpha, argv[2], TOP_TYPE);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size / sizeof(TYPE); i++)
    result_data[i] = alpha * first_data[i] / second_data[i];

  return result;
}

TYPED_NIF(dot, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(dot_and_add, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(dot_and_appply, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(dot_nt, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(dot_tn, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(eye, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  long size;
  TOP_TYPE value;
  TYPE *result_data;
  long result_byte_size;

  UNUSED_VAR(argc);

  enif_get_int64(env, argv[0], &size);
  ENIF_GET_VAL(value, argv[1], TOP_TYPE);

  result_byte_size = size * size * sizeof(TYPE);
  result_data = (TYPE*)enif_make_new_binary(env, result_byte_size, &result);

  memset((void*)result_data, 0, result_byte_size);

  for (uint64_t x = 0, y = 0; x < size && y < size; x++, y++)
    result_data[y*size + x] = (TYPE)value;

  return result;
}

TYPED_NIF(fill, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  unsigned long size;
  ErlNifBinary value;
  TYPE scalar;
  TYPE* result_data;

  UNUSED_VAR(argc);

  enif_get_uint64(env, argv[0], &size);
  if (!enif_inspect_binary(env, argv[1], &value)) return enif_make_badarg(env);
  scalar = *((TYPE*)value.data);

  result_data = (TYPE*)enif_make_new_binary(env, size * sizeof(TYPE), &result);

  for (uint64_t i = 0; i < size; i++)
    result_data[i] = scalar;

  return result;
}

TYPED_NIF(find, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, element;
  TYPE *matrix_data, *element_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &element)) return enif_make_badarg(env);

  matrix_data = (TYPE*) matrix.data;
  element_data = (TYPE*) element.data;

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    if (matrix_data[i] == *element_data)
      return enif_make_int(env, i);

  return enif_make_atom(env, "nil");
}

TYPED_NIF(from_range, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  long from, to;
  TYPE *result_data;
  size_t result_size;

  UNUSED_VAR(argc);

  enif_get_int64(env, argv[0], &from);
  enif_get_int64(env, argv[1], &to);

  result_size = to - from + 1;
  result_data = (TYPE*) enif_make_new_binary(env, result_size * sizeof(TYPE), &result);

  for (int64_t index = 0; index < result_size; index += 1)
    result_data[index] = (TYPE)(from + index);
  
  return result;
}

TYPED_NIF(max, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary matrix;
  ERL_NIF_TERM result;
  TYPE *matrix_data;
  TYPE max;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  matrix_data = (TYPE*)matrix.data;

  max = matrix_data[0];
  for (int64_t i = 1; i < matrix.size / sizeof(TYPE); i++)
    if (matrix_data[i] > max) max = matrix_data[i];

  return ENIF_MAKE_VAL(max, TOP_TYPE);
}

TYPED_NIF(max_finite, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(min, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(min_finite, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(multiply, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(multiply_with_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(neg, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(random, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  TYPE *result_data;
  unsigned long size;

  UNUSED_VAR(argc);

  enif_get_uint64(env, argv[0], &size);

  result_data = (TYPE*)enif_make_new_binary(env, size * sizeof(TYPE), &result);

  for (uint64_t i = 0; i < size; i++)
    result_data[i] = (TYPE)random() / (TYPE)RAND_MAX;

  return result;
}

TYPED_NIF(resize, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(row_to_list, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;

  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(set, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(set_column, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(submatrix, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(subtract, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(subtract_from_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(sum, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  TYPE *matrix_data;
  TOP_TYPE sum = 0;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data  = (TYPE*)matrix.data;

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    sum += matrix_data[i];

  return ENIF_MAKE_VAL(sum, TOP_TYPE);
}

TYPED_NIF(to_list, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(to_list_of_lists, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}

TYPED_NIF(transpose, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  UNUSED_VAR(argc);

  return result;
}


// NIFs defined only for floating point types (float32 & float64)
#ifdef FLOAT_NIFS

#define MATH_FUNC_TYPE(TN) CAT(math_func_, TN, _ptr_t)
#define MATH_FUNC_FROM_NAME(TN) CAT(math_func_, TN, _from_name)

TYPED_NIF(apply_math, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  char          function_name[16];
  ERL_NIF_TERM  result;
  TYPE *array_data, *result_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &array )) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[1], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(
             env,
             enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));

  MATH_FUNC_TYPE(TYPE_NAME) func = MATH_FUNC_FROM_NAME(TYPE_NAME)(function_name);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size / sizeof(TYPE); i++)
    result_data[i] = func(array_data[i]);

  return result;
}


TYPED_NIF(normalize, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;

  return result;
}

#endif
