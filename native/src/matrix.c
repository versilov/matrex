#include "../include/matrix.h"

void
matrix_clone(Matrix destination, Matrix source) {
  uint64_t length = source[0] * source[1] + 2;

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
matrix_new(int32_t rows, int32_t columns) {
  uint64_t length = rows * columns + 2;
  Matrix  result = malloc(sizeof(float) * length);

  result[0] = rows;
  result[1] = columns;

  return result;
}

void
matrix_fill(Matrix matrix, int32_t value) {
  uint64_t length = matrix[0] * matrix[1] + 2;

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = value;
  }
}

void
matrix_random(Matrix matrix) {
  uint64_t length = matrix[0] * matrix[1] + 2;

  srand(time(NULL));

  for (uint64_t index = 2; index < length; index += 1) {
    matrix[index] = (float)rand()/(float)RAND_MAX;
  }
}

void
matrix_eye(Matrix matrix, int32_t value) {
  uint64_t length = matrix[0] * matrix[1] * sizeof(float);
  int32_t rows = (int32_t)matrix[0];
  int32_t cols = (int32_t)matrix[1];

  memset((void*)&matrix[2], 0, length);
  for (int32_t x = 0, y = 0; x < cols && y < rows; x++, y++) {
    matrix[y*cols + x + 2] = value;
  }
}

void
matrix_zeros(Matrix matrix) {
  uint64_t length = matrix[0] * matrix[1] * sizeof(float);
  memset((void*)&matrix[2], 0, length);
}

int32_t
matrix_equal(Matrix first, Matrix second) {
  if (first[0] != second[0]) return 0;
  if (first[1] != second[1]) return 0;

  uint64_t length = first[0] * first[1] + 2;

  for (uint64_t index = 2; index < length; index += 1) {
    if (first[index] != second[index]) return 0;
  }

  return 1;
}

void
matrix_add(const Matrix first, const Matrix second, Matrix result) {
  uint64_t data_size = (uint64_t) (first[0] * first[1] + 2);

  result[0] = first[0];
  result[1] = first[1];

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] + second[index];
  }
}

int32_t
matrix_argmax(const Matrix matrix) {
  uint64_t data_size = (int64_t) (matrix[0] * matrix[1] + 2);
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
  uint64_t data_size = (uint64_t) (first[0] * first[1] + 2);

  result[0] = first[0];
  result[1] = first[1];

  for (uint64_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] / second[index];
  }
}

void
matrix_dot(const Matrix first, const Matrix second, Matrix result) {
  result[0] = first[0];
  result[1] = second[1];

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    first[0],
    second[1],
    first[1],
    1.0,
    first + 2,
    first[1],
    second + 2,
    second[1],
    0.0,
    result + 2,
    result[1]
  );
}

void
matrix_dot_and_add(
  const Matrix first, const Matrix second, const Matrix third, Matrix result
) {
  uint64_t data_size = (uint64_t) (first[0] * second[1] + 2);

  result[0] = first[0];
  result[1] = second[1];

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    first[0],
    second[1],
    first[1],
    1.0,
    first + 2,
    first[1],
    second + 2,
    second[1],
    0.0,
    result + 2,
    result[1]
  );

  for(uint64_t index = 2; index < data_size; index += 1) {
    result[index] += third[index];
  }
}

void
matrix_dot_nt(const Matrix first, const Matrix second, Matrix result) {
  result[0] = first[0];
  result[1] = second[0];

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasTrans,
    first[0],
    second[0],
    first[1],
    1.0,
    first + 2,
    first[1],
    second + 2,
    second[1],
    0.0,
    result + 2,
    result[1]
  );
}

void
matrix_dot_tn(const Matrix first, const Matrix second, Matrix result) {
  result[0] = first[1];
  result[1] = second[1];

  cblas_sgemm(
    CblasRowMajor,
    CblasTrans,
    CblasNoTrans,
    first[1],
    second[1],
    first[0],
    1.0,
    first + 2,
    first[1],
    second + 2,
    second[1],
    0.0,
    result + 2,
    result[1]
  );
}

float
matrix_first(const Matrix matrix) {
  return matrix[2];
}

void
matrix_inspect(const Matrix matrix) {
  int32_t length = matrix[0] * matrix[1] + 2;

  printf("<#Matrix\n");

  printf("  rows:    %f\n", matrix[0]);
  printf("  columns: %f\n", matrix[1]);

  printf("  values: ");
  for(int32_t index = 2; index < length; index += 1) {
    printf(" %f", matrix[index]);
  }

  printf(">\n");
}

void
matrix_inspect_internal(const Matrix matrix, int32_t indentation) {
  int32_t length = matrix[0] * matrix[1] + 2;

  printf("<#Matrix\n");

  print_spaces(indentation);
  printf("  rows:    %f\n", matrix[0]);

  print_spaces(indentation);
  printf("  columns: %f\n", matrix[1]);

  print_spaces(indentation);
  printf("  values: ");
  for(int32_t index = 2; index < length; index += 1) {
    printf(" %f", matrix[index]);
  }
  printf(">");
}

float
matrix_max(const Matrix matrix) {
  int32_t data_size = (int32_t) (matrix[0] * matrix[1] + 2);
  float   max       = matrix[2];

  for (int32_t index = 3; index < data_size; index += 1) {
    if (max < matrix[index]) {
      max = matrix[index];
    }
  }

  return max;
}

void
matrix_multiply(const Matrix first, const Matrix second, Matrix result) {
  int32_t data_size = (int32_t) (first[0] * first[1] + 2);

  result[0] = first[0];
  result[1] = first[1];

  for (int32_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] * second[index];
  }
}

void
matrix_multiply_with_scalar(
  const Matrix matrix, const float scalar, Matrix result
) {
  int32_t data_size = (int32_t) (matrix[0] * matrix[1] + 2);

  result[0] = matrix[0];
  result[1] = matrix[1];

  for (int32_t index = 2; index < data_size; index += 1) {
    result[index] = matrix[index] * scalar;
  }
}

void
matrix_substract(const Matrix first, const Matrix second, Matrix result) {
  int32_t data_size = (int32_t) (first[0] * first[1] + 2);

  result[0] = first[0];
  result[1] = first[1];

  for (int32_t index = 2; index < data_size; index += 1) {
    result[index] = first[index] - second[index];
  }
}

float
matrix_sum(const Matrix matrix) {
  int32_t data_size = matrix[0] * matrix[1] + 2;
  float   sum       = 0;

  for (int32_t index = 2; index < data_size; index += 1) {
    sum += matrix[index];
  }

  return sum;
}

void
matrix_transpose(const Matrix matrix, Matrix result) {
  result[0] = matrix[1];
  result[1] = matrix[0];

  for (int32_t row = 0; row < matrix[0]; row += 1) {
    for (int32_t column = 0; column < matrix[1]; column += 1) {
      int32_t result_index = column * result[1] + row    + 2;
      int32_t matrix_index = row *    matrix[1] + column + 2;

      result[result_index] = matrix[matrix_index];
    }
  }
}
