#include "../include/matrix.h"

void
matrix_clone(Matrix destination, Matrix source) {
  uint64_t length = MX_LENGTH(source);

  for (uint64_t index = 0; index < length; index += 1) {
    destination[index] = source[index];
  }
}

void
matrix_free(Matrix *matrix_address) {
  Matrix matrix = *matrix_address;

  if (matrix != NULL) free(matrix);

  *matrix_address = NULL;
}

Matrix
matrix_new(uint32_t rows, uint32_t columns) {
  uint64_t length = rows * columns + 2;
  Matrix  result = malloc(sizeof(float) * length);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, columns);

  return result;
}

int32_t
matrix_equal(const Matrix first, const Matrix second) {
  if (MX_ROWS(first) != MX_ROWS(second)) return 0;
  if (MX_COLS(first) != MX_COLS(second)) return 0;

  uint64_t length = MX_LENGTH(first);

  for (uint64_t index = 2; index < length; index += 1) {
    if (first[index] != second[index]) return 0;
  }

  return 1;
}

void
matrix_add(const Matrix first, const Matrix second,
  const float alpha, const float beta, Matrix result) {
    uint64_t data_size = MX_LENGTH(first);

    MX_SET_ROWS(result, MX_ROWS(first));
    MX_SET_COLS(result, MX_COLS(first));

    for (uint64_t index = 2; index < data_size; index += 1) {
      result[index] = alpha*first[index] + beta*second[index];
    }
}


void
matrix_add_scalar(
  const Matrix matrix, const float scalar, Matrix result
) {
  uint64_t data_size = MX_LENGTH(matrix);

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = matrix[index] + scalar;
  }
}

float sigmoidf(float x) {
  return 1.0f/(1.0f + expf(-x));
}

math_func_ptr_t math_func_from_name(const char* name) {
  if (strcmp(name, "exp") == 0)
    return &expf;
  if (strcmp(name, "exp2") == 0)
    return &exp2f;
  if (strcmp(name, "sigmoid") == 0)
    return &sigmoidf;
  if (strcmp(name, "expm1") == 0)
    return &expm1f;
  if (strcmp(name, "ceil") == 0)
    return &ceilf;
  if (strcmp(name, "floor") == 0)
    return &floorf;
  if (strcmp(name, "truncate") == 0 || strcmp(name, "trunc") == 0)
    return &truncf;
  if (strcmp(name, "round") == 0)
    return &roundf;
  if (strcmp(name, "abs") == 0)
    return &fabsf;
  if (strcmp(name, "erf") == 0)
    return &erff;
  if (strcmp(name, "erfc") == 0)
    return &erfcf;
  if (strcmp(name, "tgamma") == 0)
    return &tgammaf;
  if (strcmp(name, "lgamma") == 0)
    return &lgammaf;
  if (strcmp(name, "log") == 0)
    return &logf;
  if (strcmp(name, "log2") == 0)
    return &log2f;
  if (strcmp(name, "sqrt") == 0)
    return &sqrtf;
  if (strcmp(name, "cbrt") == 0)
    return &cbrtf;
  if (strcmp(name, "sin") == 0)
    return &sinf;
  if (strcmp(name, "cos") == 0)
    return &cosf;
  if (strcmp(name, "tan") == 0)
    return &tanf;
  if (strcmp(name, "asin") == 0)
    return &asinf;
  if (strcmp(name, "acos") == 0)
    return &acosf;
  if (strcmp(name, "atan") == 0)
    return &atanf;
  if (strcmp(name, "sinh") == 0)
    return &sinhf;
  if (strcmp(name, "cosh") == 0)
    return &coshf;
  if (strcmp(name, "tanh") == 0)
    return &tanhf;
  if (strcmp(name, "asinh") == 0)
    return &asinhf;
  if (strcmp(name, "acosh") == 0)
    return &acoshf;
  if (strcmp(name, "atanh") == 0)
    return &atanhf;
  return NULL;
}

int
matrix_apply(const Matrix matrix, char* function_name, Matrix result) {
  const uint64_t data_size = MX_LENGTH(matrix);
  const math_func_ptr_t func = math_func_from_name(function_name);

  if (func == NULL) return 0;

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = func(matrix[index]);
  }

  return 1;
}

int32_t
matrix_argmax(const Matrix matrix) {
  const int64_t data_size = MX_LENGTH(matrix);
  uint64_t argmax    = 2;

  for (int64_t index = 3; index < data_size; index += 1) {
    if (matrix[argmax] < matrix[index]) {
      argmax = index;
    }
  }
  return argmax - 2;
}


void
matrix_concat_columns(const Matrix first, const Matrix second, Matrix result) {
  const int64_t result_cols = MX_COLS(first) + MX_COLS(second);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, result_cols);

  for (int64_t row = 0; row < MX_ROWS(first); row++) {
    memcpy(&result[2 + row*result_cols], &first[2 + row*MX_COLS(first)], MX_COLS(first)*sizeof(float));
    memcpy(&result[2 + row*result_cols + MX_COLS(first)], &second[2 + row*MX_COLS(second)], MX_COLS(second)*sizeof(float));
  }
}


void
matrix_divide(const Matrix first, const Matrix second, Matrix result) {
  const int64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (int64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] / second[index];
  }
}

void
matrix_divide_scalar(const float scalar, const Matrix divisor, Matrix result) {
  const int64_t data_size = MX_LENGTH(divisor);

  MX_SET_ROWS(result, MX_ROWS(divisor));
  MX_SET_COLS(result, MX_COLS(divisor));

  for (int64_t index = 2; index < data_size; index += 1) {
    result[index] = scalar / divisor[index];
  }
}

void
matrix_divide_by_scalar(const Matrix dividend, const float scalar, Matrix result) {
  const uint64_t data_size = MX_LENGTH(dividend);

  MX_SET_ROWS(result, MX_ROWS(dividend));
  MX_SET_COLS(result, MX_COLS(dividend));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = dividend[index] / scalar;
  }
}

void
matrix_eye(Matrix matrix, const float value) {
  const uint64_t length = MX_DATA_BYTE_SIZE(matrix);
  const uint64_t rows = MX_ROWS(matrix);
  const uint64_t cols = MX_COLS(matrix);

  // Set it all to zeros
  memset((void*)&matrix[2], 0, length);

  // Now set the diagonal
  for (uint64_t x = 0, y = 0; x < cols && y < rows; x++, y++) {
    matrix[2 + y*cols + x] = value;
  }
}

void
matrix_diagonal(const Matrix matrix, const uint64_t diag_size, Matrix result) {
  // Set it all to zeros
  const uint64_t cols = MX_COLS(matrix);

  // Now set the diagonal
  for (uint64_t i = 0; i < diag_size; i++) {
    result[2 + i] = matrix[2 + i*cols + i];
  }
}

void
matrix_fill(Matrix matrix, const float value) {
  const uint64_t length = MX_LENGTH(matrix);

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = value;
  }
}

int32_t
matrix_find(const Matrix matrix, const float value) {
  uint64_t data_size = MX_LENGTH(matrix);

  for (uint64_t index = 2; index < data_size; index += 1) {
    if (matrix[index] == value) return index - 2;
  }
  return -1;
}

int32_t
matrix_find_nan(const Matrix matrix) {
  const int64_t data_size = MX_LENGTH(matrix);

  for (int64_t index = 2; index < data_size; index += 1) {
    if (isnan(matrix[index])) return index - 2;
  }
  return -1;
}

float
matrix_first(const Matrix matrix) {
  return matrix[2];
}

void
matrix_from_range(const int64_t from, const int64_t to, const int64_t rows, const int64_t cols, Matrix result) {
  const int64_t data_size = rows*cols + 2;

  (void)(to);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t index = 2; index < data_size; index += 1) {
    result[index] = from + index - 2;
  }
}

void
matrix_inspect(const Matrix matrix) {
  uint64_t length = MX_LENGTH(matrix);

  printf("<#Matrix\n");

  printf("  rows:    %d\n", MX_ROWS(matrix));
  printf("  columns: %d\n", MX_COLS(matrix));

  printf("  values: ");
  for(uint64_t index = 2; index < length; index += 1) {
    printf(" %f", matrix[index]);
  }

  printf(">\n");
}

void
matrix_inspect_internal(const Matrix matrix, int32_t indentation) {
  uint64_t length = MX_LENGTH(matrix);

  printf("<#Matrix\n");

  print_spaces(indentation);
  printf("  rows:    %d\n", MX_ROWS(matrix));

  print_spaces(indentation);
  printf("  columns: %d\n", MX_COLS(matrix));

  print_spaces(indentation);
  printf("  values: ");
  for(uint64_t index = 2; index < length; index += 1) {
    printf(" %f", matrix[index]);
  }
  printf(">");
}

float
matrix_max(const Matrix matrix) {
  const uint64_t data_size = MX_LENGTH(matrix);
  float   max       = matrix[2];

  for (uint64_t index = 3; index < data_size; index += 1) {
    if (max < matrix[index]) {
      max = matrix[index];
    }
  }

  return max;
}

float
matrix_min(const Matrix matrix) {
  const uint64_t data_size = MX_LENGTH(matrix);
  float   min       = matrix[2];

  for (uint64_t index = 3; index < data_size; index += 1) {
    if (min > matrix[index]) {
      min = matrix[index];
    }
  }

  return min;
}

float
matrix_max_finite(const Matrix matrix) {
  const uint64_t data_size = MX_LENGTH(matrix);
  float   max       = NAN;

  for (uint64_t index = 2; index < data_size; index += 1) {
    if (isfinite(matrix[index]) && (isnan(max) || max < matrix[index])) {
      max = matrix[index];
    }
  }

  return max;
}

float
matrix_min_finite(const Matrix matrix) {
  const uint64_t data_size = MX_LENGTH(matrix);
  float   min       = NAN;

  for (uint64_t index = 2; index < data_size; index += 1) {
    if (isfinite(matrix[index]) && (isnan(min) || min > matrix[index])) {
      min = matrix[index];
    }
  }

  return min;
}

void
matrix_multiply(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] * second[index];
  }
}

void
matrix_multiply_with_scalar(
  const Matrix matrix, const float scalar, Matrix result
) {
  uint64_t data_size = MX_LENGTH(matrix);

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = matrix[index] * scalar;
  }
}

void
matrix_neg(
  const Matrix matrix, Matrix result
) {
  uint64_t data_size = MX_LENGTH(matrix);

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = -matrix[index];
  }
}

void
matrix_normalize(const Matrix matrix, Matrix result) {
  uint64_t data_size = MX_LENGTH(matrix);
  float min = matrix_min(matrix);
  float max = matrix_max(matrix);
  float range = max - min;

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = (matrix[index] - min)/range;
  }
}


void
matrix_random(Matrix matrix) {
  uint64_t length = MX_LENGTH(matrix);

  // RNG is initialized in ELR_NIF_INIT load function.

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = (float)random()/(float)RAND_MAX;
  }
}

void
matrix_resize(const Matrix matrix, const int32_t new_rows, const int32_t new_cols, Matrix result) {
  MX_SET_ROWS(result, new_rows);
  MX_SET_COLS(result, new_cols);

  const double row_scale = (double)new_rows / (double)MX_ROWS(matrix);
  const double col_scale = (double)new_cols / (double)MX_COLS(matrix);

  for (int32_t row = 0; row < new_rows; row++)
    for (int32_t col = 0; col < new_cols; col++)
      result[2 + row*new_cols + col] = matrix[2 + (int)trunc((double)row/row_scale)*MX_COLS(matrix) + (int)trunc((double)col/col_scale)];
}


void
matrix_set(const Matrix matrix, const uint32_t row, const uint32_t column, const float scalar, Matrix result) {
  uint64_t data_size = MX_BYTE_SIZE(matrix);


  memcpy(result, matrix, data_size);
  // cblas_scopy(MX_LENGTH(matrix), matrix, 1, result, 1);
  result[2 + row*MX_COLS(matrix) + column] = scalar;
}

void
matrix_set_column(const Matrix matrix, const uint32_t column, const Matrix column_matrix, Matrix result) {
  memcpy(result, matrix, MX_BYTE_SIZE(matrix));
  // cblas_scopy(MX_LENGTH(matrix), matrix, 1, result, 1);
  for (uint64_t row = 0; row < MX_ROWS(matrix); row++ )
    result[2 + row*MX_COLS(matrix) + column] = column_matrix[2 + row];
}


void
matrix_submatrix(const Matrix matrix, const uint32_t row_from, const uint32_t row_to,
  const uint32_t column_from, const uint32_t column_to, Matrix result) {

  const uint32_t source_columns = MX_COLS(matrix);

  const uint32_t rows = row_to - row_from + 1;
  const uint32_t columns = column_to - column_from + 1;

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, columns);

  for (uint32_t row = row_from; row <= row_to; row++)
    memcpy(&result[2 + (row - row_from)*columns],
           &matrix[2 + row*source_columns + column_from],
           columns * sizeof(float));
    // cblas_scopy(columns, &matrix[2 + row*source_columns + column_from], 1,
    //                      &result[2 + (row - row_from)*columns], 1);
}

void
matrix_subtract(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] - second[index];
  }
}

void
matrix_subtract_from_scalar(const float scalar, const Matrix matrix, Matrix result) {
  uint64_t data_size = MX_LENGTH(matrix);

  MX_SET_ROWS(result, MX_ROWS(matrix));
  MX_SET_COLS(result, MX_COLS(matrix));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = scalar - matrix[index];
  }
}

double
matrix_sum(const Matrix matrix) {
  uint64_t data_size = MX_LENGTH(matrix);
  double   sum       = 0.0;

  for (uint64_t index = 2; index < data_size; index += 1) {
    sum += matrix[index];
  }

  return sum;
}

void
matrix_transpose(const Matrix matrix, Matrix result) {
  MX_SET_ROWS(result, MX_COLS(matrix));
  MX_SET_COLS(result, MX_ROWS(matrix));

  for (uint64_t row = 0; row < MX_ROWS(matrix); row += 1) {
    for (uint64_t column = 0; column < MX_COLS(matrix); column += 1) {
      uint64_t result_index = column * MX_COLS(result) + row    + 2;
      uint64_t matrix_index = row *    MX_COLS(matrix) + column + 2;

      result[result_index] = matrix[matrix_index];
    }
  }
}


void
matrix_zeros(Matrix matrix) {
  uint64_t length = MX_DATA_BYTE_SIZE(matrix);
  memset((void*)&matrix[2], 0, length);
}
