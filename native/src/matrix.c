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

void
matrix_fill(Matrix matrix, int32_t value) {
  uint64_t length = MX_LENGTH(matrix);

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = value;
  }
}

void
matrix_random(Matrix matrix) {
  uint64_t length = MX_LENGTH(matrix);

  srandom(time(NULL));

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = (float)random()/(float)RAND_MAX;
  }
}

void
matrix_eye(Matrix matrix, int32_t value) {
  uint64_t length = MX_DATA_BYTE_SIZE(matrix);
  uint64_t rows = MX_ROWS(matrix);
  uint64_t cols = MX_COLS(matrix);

  // Set it all to zeros
  memset((void*)&matrix[2], 0, length);

  // Now set the diagonal
  for (uint64_t x = 0, y = 0; x < cols && y < rows; x++, y++) {
    matrix[2 + y*cols + x] = value;
  }
}

void
matrix_zeros(Matrix matrix) {
  uint64_t length = MX_DATA_BYTE_SIZE(matrix);
  memset((void*)&matrix[2], 0, length);
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
matrix_add(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] + second[index];
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

math_func_ptr_t math_func_from_name(char* name) {
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
  if (strcmp(name, "trunc") == 0)
    return &truncf;
  if (strcmp(name, "round") == 0)
    return &roundf;
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
  uint64_t data_size = MX_LENGTH(matrix);
  math_func_ptr_t func = math_func_from_name(function_name);

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
  uint64_t data_size = MX_LENGTH(matrix);
  uint64_t argmax    = 2;

  for (uint64_t index = 3; index < data_size; index += 1) {
    if (matrix[argmax] < matrix[index]) {
      argmax = index;
    }
  }
  return argmax - 2;
}

void
matrix_divide(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] / second[index];
  }
}

void
matrix_dot(const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MX_ROWS(first),
    MX_COLS(second),
    MX_COLS(first),
    1.0,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );
}

void
matrix_dot_and_add(
  const Matrix first, const Matrix second, const Matrix third, Matrix result
) {
  uint64_t data_size = MX_ROWS(first) * MX_COLS(second) + 2;

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MX_ROWS(first),
    MX_COLS(second),
    MX_COLS(first),
    1.0,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );

  for(uint64_t index = 2; index < data_size; index += 1) {
    result[index] += third[index];
  }
}

void
matrix_dot_nt(const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_ROWS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasTrans,
    MX_ROWS(first),
    MX_ROWS(second),
    MX_COLS(first),
    1.0,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );
}

void
matrix_dot_tn(const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_COLS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasTrans,
    CblasNoTrans,
    MX_COLS(first),
    MX_COLS(second),
    MX_ROWS(first),
    1.0,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );
}

float
matrix_first(const Matrix matrix) {
  return matrix[2];
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
  uint64_t data_size = MX_LENGTH(matrix);
  float   max       = matrix[2];

  for (uint64_t index = 3; index < data_size; index += 1) {
    if (max < matrix[index]) {
      max = matrix[index];
    }
  }

  return max;
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
matrix_substract(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = MX_LENGTH(first);

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(first));

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] - second[index];
  }
}

void
matrix_substract_from_scalar(const float scalar, const Matrix matrix, Matrix result) {
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
