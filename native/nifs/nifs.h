TYPED_NIF(add, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  TYPE *first_data, *second_data, *result_data;
  TOP_TYPE alpha, beta;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  ENIF_GET_VAL(alpha, argv[2], TOP_TYPE);
  ENIF_GET_VAL(beta, argv[3], TOP_TYPE);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size / sizeof(TYPE); i++)
    result_data[i] = alpha*first_data[i] + beta*second_data[i];

  return result;
}

TYPED_NIF(add_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  ERL_NIF_TERM  result;
  TYPE        *array_data, *result_data;
  TOP_TYPE scalar, alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &array)) return enif_make_badarg(env);
  ENIF_GET_VAL(scalar, argv[1], TOP_TYPE);
  ENIF_GET_VAL(alpha, argv[2], TOP_TYPE);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size / sizeof(TYPE); i++)
    result_data[i] = alpha*array_data[i] + scalar;

  return result;
}


TYPED_NIF(argmax, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary matrix;
  TYPE *matrix_data;
  TYPE max;
  int64_t argmax = 0;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  matrix_data = (TYPE*)matrix.data;

  max = matrix_data[0];
  for (int64_t i = 1; i < matrix.size / sizeof(TYPE); i++)
    if (matrix_data[i] > max) {
      max = matrix_data[i];
      argmax = i;
    } 

  return enif_make_int64(env, argmax);
}

TYPED_NIF(column_to_list, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  TYPE *matrix_data;
  long rows, cols, column;
  ERL_NIF_TERM  result;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_int64(env, argv[1], &cols);
  enif_get_int64(env, argv[2], &column);

  rows = matrix.size / (cols * sizeof(TYPE));

  matrix_data = (TYPE*) matrix.data;

  if (column >= cols)
    return enif_raise_exception(env, enif_make_string(env, "Column index out of bounds.", ERL_NIF_LATIN1));

  result = enif_make_list(env, 0);

  for (int64_t i = (rows-1)*cols + column ; i >= column; i -= cols ) {
    result = enif_make_list_cell(env, ENIF_MAKE_VAL(matrix_data[i], TOP_TYPE), result);
  }

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
  ErlNifBinary  matrix;
  TYPE *matrix_data, *result_data;
  TOP_TYPE scalar;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);
  ENIF_GET_VAL(scalar, argv[1], TOP_TYPE);

  matrix_data  = (TYPE*)matrix.data;

  result_data = (TYPE*)enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = matrix_data[i] / scalar;

  return result;
}

TYPED_NIF(divide_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  ErlNifBinary  matrix;
  TYPE *matrix_data, *result_data;
  TOP_TYPE scalar;

  UNUSED_VAR(argc);

  ENIF_GET_VAL(scalar, argv[0], TOP_TYPE);
  if (!enif_inspect_binary(env, argv[1], &matrix )) return enif_make_badarg(env);

  matrix_data  = (TYPE*)matrix.data;

  result_data = (TYPE*)enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = scalar / matrix_data[i];

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

#ifdef BLAS_GEMM

TYPED_NIF(dot, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
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
  get_scalar_double(env, argv[5], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, rows*cols*sizeof(TYPE), &result);

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

TYPED_NIF(dot_and_add, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second, third;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *third_data, *result_data;
  long rows, dim, cols;
  double alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &dim);
  enif_get_int64(env, argv[4], &cols);
  if (!enif_inspect_binary(env, argv[5], &third)) return enif_make_badarg(env);
  get_scalar_double(env, argv[6], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;
  third_data = (TYPE*)third.data;

  result_data = (TYPE*)enif_make_new_binary(env, rows*cols*sizeof(TYPE), &result);

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

  for(uint64_t index = 0; index < rows*cols; index += 1) {
    result_data[index] += third_data[index];
  }

  return result;
}

TYPED_NIF(dot_and_apply, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *result_data;
  long rows, dim, cols;
  char function_name[16];
  double alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &dim);
  enif_get_int64(env, argv[4], &cols);
  if (enif_get_atom(env, argv[5], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(env, enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));
  get_scalar_double(env, argv[6], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  MATH_FUNC_TYPE(TYPE_NAME) func = MATH_FUNC_FROM_NAME(TYPE_NAME)(function_name);

  result_data = (TYPE*)enif_make_new_binary(env, rows*cols*sizeof(TYPE), &result);

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

  for(uint64_t index = 0; index < rows*cols; index += 1) {
    result_data[index] = func(result_data[index]);
  }

  return result;
}

TYPED_NIF(dot_nt, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *result_data;
  long rows, dim, cols;
  double alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &dim);
  enif_get_int64(env, argv[4], &cols);
  get_scalar_double(env, argv[5], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, rows*cols*sizeof(TYPE), &result);

  BLAS_GEMM(
    CblasRowMajor,// Order — Specifies row-major (C) or column-major (Fortran) data ordering.
    CblasNoTrans,   // TransA — Specifies whether to transpose matrix A.
    CblasTrans, // TransB — Specifies whether to transpose matrix B.
    rows,         // M — Number of rows in matrices A and C.
    cols,         // N — Number of columns in matrices B and C.
    dim,          // K — Number of columns in matrix A; number of rows in matrix B.
    alpha,        // alpha — Scaling factor for the product of matrices A and B.
    first_data,   // A — Matrix A.
    dim,         // lda — The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
    second_data,  // B — Matrix B.
    dim,         // ldb — The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.
    0.0,          // beta — Scaling factor for matrix C.
    result_data,  // C — Matrix C.
    cols          // ldc — The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.
  );

  return result;
}

TYPED_NIF(dot_tn, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary first, second;
  ERL_NIF_TERM result;
  TYPE *first_data, *second_data, *result_data;
  long rows, dim, cols;
  double alpha;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  enif_get_int64(env, argv[2], &rows);
  enif_get_int64(env, argv[3], &dim);
  enif_get_int64(env, argv[4], &cols);
  get_scalar_double(env, argv[5], &alpha);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, rows*cols*sizeof(TYPE), &result);

  BLAS_GEMM(
    CblasRowMajor,// Order — Specifies row-major (C) or column-major (Fortran) data ordering.
    CblasTrans,   // TransA — Specifies whether to transpose matrix A.
    CblasNoTrans, // TransB — Specifies whether to transpose matrix B.
    rows,         // M — Number of rows in matrices A and C.
    cols,         // N — Number of columns in matrices B and C.
    dim,          // K — Number of columns in matrix A; number of rows in matrix B.
    alpha,        // alpha — Scaling factor for the product of matrices A and B.
    first_data,   // A — Matrix A.
    rows,         // lda — The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
    second_data,  // B — Matrix B.
    cols,         // ldb — The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.
    0.0,          // beta — Scaling factor for matrix C.
    result_data,  // C — Matrix C.
    cols          // ldc — The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.
  );

  return result;
}

#else

// Implement naive integer matrix dot

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


#endif


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

  for (int64_t x = 0, y = 0; x < size && y < size; x++, y++)
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

  if (isnan(*element_data)) {
    for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
      if (isnan(matrix_data[i]))
        return enif_make_int(env, i);
  } else {
    for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
      if (matrix_data[i] == *element_data)
        return enif_make_int(env, i);
  }

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

TYPED_NIF(min, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary matrix;
  TYPE *matrix_data;
  TYPE min;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  matrix_data = (TYPE*)matrix.data;

  min = matrix_data[0];
  for (int64_t i = 1; i < matrix.size / sizeof(TYPE); i++)
    if (matrix_data[i] < min) min = matrix_data[i];

  return ENIF_MAKE_VAL(min, TOP_TYPE);
}

TYPED_NIF(multiply, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
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
    result_data[i] = alpha * first_data[i] * second_data[i];

  return result;
}

TYPED_NIF(multiply_with_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  ErlNifBinary  matrix;
  TYPE *matrix_data, *result_data;
  TOP_TYPE scalar;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);
  ENIF_GET_VAL(scalar, argv[1], TOP_TYPE);

  matrix_data  = (TYPE*)matrix.data;

  result_data = (TYPE*)enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = scalar * matrix_data[i];

  return result;
}

TYPED_NIF(neg, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ERL_NIF_TERM result;
  ErlNifBinary  matrix;
  TYPE *matrix_data, *result_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);

  matrix_data  = (TYPE*)matrix.data;

  result_data = (TYPE*)enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = -matrix_data[i];

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
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  double  scale;
  int64_t rows, cols, new_rows, new_cols;
  TYPE  *matrix_data, *result_data;
  size_t result_size;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_int64(env, argv[1], &rows);
  enif_get_int64(env, argv[2], &cols);
  get_scalar_double(env, argv[3], &scale);

  matrix_data = (TYPE *) matrix.data;

  new_rows = (int64_t)round((double)rows * scale);
  new_cols = (int64_t)round((double)cols * scale);

  result_size = sizeof(TYPE) * new_rows * new_cols;
  result_data = (TYPE *) enif_make_new_binary(env, result_size, &result);

  for (int64_t row = 0; row < new_rows; row++)
    for (int64_t col = 0; col < new_cols; col++)
      result_data[row*new_cols + col] =
        matrix_data[(int)trunc((double)row/scale)*cols + (int)trunc((double)col/scale)];

  return result;
}

TYPED_NIF(row_to_list, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  TYPE *matrix_data;
  long columns, row;
  ERL_NIF_TERM  result;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_int64(env, argv[1], &columns);
  enif_get_int64(env, argv[2], &row);

  matrix_data = (TYPE *) matrix.data;

  result = enif_make_list(env, 0);

  for (int64_t i = (row + 1)*columns; i-- > row*columns; )
    result = enif_make_list_cell(env, ENIF_MAKE_VAL(matrix_data[i], TOP_TYPE), result);

  return result;

}

TYPED_NIF(set, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, value;
  TYPE *matrix_data, *result_data;
  unsigned long offset;
  TYPE scalar;
  ERL_NIF_TERM result;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &offset);
  enif_inspect_binary(env, argv[2], &value);

  scalar = *((TYPE*)value.data);

  matrix_data = (TYPE*)matrix.data;

  if (offset >= matrix.size)
    return enif_raise_exception(env, enif_make_string(env, "Position out of bounds.", ERL_NIF_LATIN1));

  result_data = (TYPE*) enif_make_new_binary(env, matrix.size, &result);

  memcpy(result_data, matrix_data, matrix.size);
  result_data[offset/sizeof(TYPE)] = scalar;

  return result;
}

TYPED_NIF(set_column, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix, column_matrix;
  TYPE *matrix_data, *column_matrix_data, *result_data;
  uint64_t  result_size;
  unsigned long column, rows, cols;
  ERL_NIF_TERM  result;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &cols);
  enif_get_uint64(env, argv[2], &column);
  if (!enif_inspect_binary(env, argv[3], &column_matrix)) return enif_make_badarg(env);


  matrix_data = (TYPE *) matrix.data;
  column_matrix_data = (TYPE *) column_matrix.data;

  if (column >= cols)
    return enif_raise_exception(env, enif_make_string(env, "Position out of bounds.", ERL_NIF_LATIN1));

  result_data = (TYPE *) enif_make_new_binary(env, matrix.size, &result);
  rows = matrix.size / (cols*sizeof(TYPE));

  memcpy(result_data, matrix_data, matrix.size);

  for (uint64_t row = 0; row < rows; row++ )
    result_data[row*cols + column] = column_matrix_data[row];

  return result;
}

TYPED_NIF(submatrix, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
ErlNifBinary  matrix;
  TYPE *matrix_data, *result_data;
  uint64_t  result_size;
  unsigned long source_columns, row_from, row_to, column_from, column_to;
  ERL_NIF_TERM  result;
  uint64_t rows, cols;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &source_columns);
  enif_get_uint64(env, argv[2], &row_from);
  enif_get_uint64(env, argv[3], &row_to);
  enif_get_uint64(env, argv[4], &column_from);
  enif_get_uint64(env, argv[5], &column_to);


  matrix_data = (TYPE *) matrix.data;
  rows = MX_ROWS(matrix_data);
  cols = MX_COLS(matrix_data);

  if (row_from >= rows || column_from >= cols || row_to >= rows || column_to >= cols)
    return enif_raise_exception(env, enif_make_string(env,
      "Submatrix position out of bounds.", ERL_NIF_LATIN1));


  rows = row_to - row_from + 1;
  cols = column_to - column_from + 1;
  result_size = rows * cols  * sizeof(TYPE);
  result_data = (TYPE *) enif_make_new_binary(env, result_size, &result);

  for (uint32_t row = row_from; row <= row_to; row++)
    memcpy(&result_data[(row - row_from)*cols],
           &matrix_data[row*source_columns + column_from],
           cols * sizeof(TYPE));

  return result;
}

TYPED_NIF(subtract, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  first, second;
  ERL_NIF_TERM  result;
  TYPE *first_data, *second_data, *result_data;
  TOP_TYPE alpha, beta;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &first )) return enif_make_badarg(env);
  if (!enif_inspect_binary(env, argv[1], &second)) return enif_make_badarg(env);
  ENIF_GET_VAL(alpha, argv[2], TOP_TYPE);
  ENIF_GET_VAL(beta, argv[3], TOP_TYPE);

  first_data  = (TYPE*)first.data;
  second_data = (TYPE*)second.data;

  result_data = (TYPE*)enif_make_new_binary(env, first.size, &result);

  for (uint64_t i = 0; i < first.size / sizeof(TYPE); i++)
    result_data[i] = alpha*first_data[i] - beta*second_data[i];

  return result;
}

TYPED_NIF(subtract_from_scalar, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  array;
  ERL_NIF_TERM  result;
  TYPE        *array_data, *result_data;
  TOP_TYPE scalar, alpha;

  UNUSED_VAR(argc);

  ENIF_GET_VAL(scalar, argv[0], TOP_TYPE);
  if (!enif_inspect_binary(env, argv[1], &array)) return enif_make_badarg(env);
  ENIF_GET_VAL(alpha, argv[2], TOP_TYPE);

  array_data  = (TYPE*)array.data;

  result_data = (TYPE*)enif_make_new_binary(env, array.size, &result);

  for (uint64_t i = 0; i < array.size / sizeof(TYPE); i++)
    result_data[i] = scalar - alpha*array_data[i];

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
  ErlNifBinary  matrix;
  TYPE *matrix_data;
  ERL_NIF_TERM  result;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (TYPE *) matrix.data;

  result = enif_make_list(env, 0);
  for (uint64_t i = matrix.size / sizeof(TYPE); i-- > 0; ) {
    result = enif_make_list_cell(env, ENIF_MAKE_VAL(matrix_data[i], TOP_TYPE), result);
  }

  return result;
}

TYPED_NIF(to_list_of_lists, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  /* to_list_of_lists(matrix) -> [[first row], [second row], ...,[last row]] */
  ErlNifBinary  matrix;
  TYPE *matrix_data;
  ERL_NIF_TERM  result;
  unsigned long rows, cols;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &rows);
  enif_get_uint64(env, argv[2], &cols);

  matrix_data = (TYPE *) matrix.data;

  result = enif_make_list(env, 0);

  for (uint64_t r = rows; r-- > 0; ) {
    ERL_NIF_TERM row = enif_make_list(env, 0);
    for (uint64_t c = cols; c-- > 0; ) {
      row = enif_make_list_cell(env, ENIF_MAKE_VAL(matrix_data[cols*r + c], TOP_TYPE), row);
    }
    result = enif_make_list_cell(env, row, result);
  }

  return result;
}

TYPED_NIF(transpose, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  TYPE *matrix_data, *result_data;
  unsigned long rows, cols;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);
  enif_get_uint64(env, argv[1], &rows);
  enif_get_uint64(env, argv[2], &cols);

  matrix_data = (TYPE *) matrix.data;
  result_data = (TYPE *) enif_make_new_binary(env, matrix.size, &result);


  for (uint64_t row = 0; row < rows; row += 1)
    for (uint64_t column = 0; column < cols; column += 1) {
      uint64_t result_index = column * rows + row;
      uint64_t matrix_index = row * cols + column;

      result_data[result_index] = matrix_data[matrix_index];
    }

  return result;
}


// NIFs defined only for floating point types (float32 & float64)
#ifdef FLOAT_NIFS

TYPED_NIF(apply_math, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  char          function_name[16];
  ERL_NIF_TERM  result;
  TYPE *matrix_data, *result_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix )) return enif_make_badarg(env);
  if (enif_get_atom(env, argv[1], function_name, 16, ERL_NIF_LATIN1) == 0)
    return enif_raise_exception(
             env,
             enif_make_string(env, "Second argument must be an atom.", ERL_NIF_LATIN1));

  MATH_FUNC_TYPE(TYPE_NAME) func = MATH_FUNC_FROM_NAME(TYPE_NAME)(function_name);

  matrix_data  = (TYPE*)matrix.data;

  result_data = (TYPE*)enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = func(matrix_data[i]);

  return result;
}

TYPED_NIF(max_finite, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  TYPE max = NAN;
  TYPE *matrix_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (TYPE*) matrix.data;

  for (uint64_t index = 0; index < matrix.size/sizeof(TYPE); index += 1) {
    if (isfinite(matrix_data[index]) && (isnan(max) || max < matrix_data[index])) {
      max = matrix_data[index];
    }
  }

  if (isnan(max))
    return enif_make_atom(env, "nil");
  else
    return enif_make_double(env, max);
}

TYPED_NIF(min_finite, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  TYPE min = NAN;
  TYPE *matrix_data;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (TYPE*) matrix.data;

  for (uint64_t index = 0; index < matrix.size/sizeof(TYPE); index += 1) {
    if (isfinite(matrix_data[index]) && (isnan(min) || min > matrix_data[index])) {
      min = matrix_data[index];
    }
  }

  if (isnan(min))
    return enif_make_atom(env, "nil");
  else
    return enif_make_double(env, min);
}

TYPED_NIF(normalize, TYPE_NAME)(ErlNifEnv *env, int32_t argc, const ERL_NIF_TERM *argv) {
  ErlNifBinary  matrix;
  ERL_NIF_TERM  result;
  TYPE *matrix_data, *result_data;
  TYPE min, max, range;

  UNUSED_VAR(argc);

  if (!enif_inspect_binary(env, argv[0], &matrix)) return enif_make_badarg(env);

  matrix_data = (TYPE*)matrix.data;

  min = matrix_data[0]; max = matrix_data[0];

  for (uint64_t i = 1; i < matrix.size / sizeof(TYPE); i++) {
    if (matrix_data[i] < min) min = matrix_data[i];
    if (matrix_data[i] > max) max = matrix_data[i];
  }
  range = max - min;

  result_data = (TYPE*) enif_make_new_binary(env, matrix.size, &result);

  for (uint64_t i = 0; i < matrix.size / sizeof(TYPE); i++)
    result_data[i] = (matrix_data[i] - min) / range;

  return result;
}

#endif
