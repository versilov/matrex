#include "../include/matrix.h"

#ifndef MATREX_NO_BLAS

#include <cblas.h>

void
matrix_dot(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MX_ROWS(first),
    MX_COLS(second),
    MX_COLS(first),
    alpha,
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
  const float alpha, const Matrix first, const Matrix second, const Matrix third, Matrix result
) {
  const uint64_t data_size = MX_ROWS(first) * MX_COLS(second) + 2;

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MX_ROWS(first),
    MX_COLS(second),
    MX_COLS(first),
    alpha,
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
matrix_dot_and_apply(
  const float alpha, const Matrix first, const Matrix second, const char *function_name, Matrix result
) {
  const math_func_ptr_t func = math_func_from_name(function_name);

  const uint64_t data_size = MX_ROWS(first) * MX_COLS(second) + 2;

  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    MX_ROWS(first),
    MX_COLS(second),
    MX_COLS(first),
    alpha,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );

  for(uint64_t index = 2; index < data_size; index += 1) {
    result[index] = func(result[index]);
  }
}


void
matrix_dot_nt(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_ROWS(first));
  MX_SET_COLS(result, MX_ROWS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasTrans,
    MX_ROWS(first),
    MX_ROWS(second),
    MX_COLS(first),
    alpha,
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
matrix_dot_tn(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  MX_SET_ROWS(result, MX_COLS(first));
  MX_SET_COLS(result, MX_COLS(second));

  cblas_sgemm(
    CblasRowMajor,
    CblasTrans,
    CblasNoTrans,
    MX_COLS(first),
    MX_COLS(second),
    MX_ROWS(first),
    alpha,
    first + 2,
    MX_COLS(first),
    second + 2,
    MX_COLS(second),
    0.0,
    result + 2,
    MX_COLS(result)
  );
}

#else

void
matrix_dot(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  const int64_t rows = MX_ROWS(first);
  const int64_t cols = MX_COLS(second);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t r = 0; r < rows; r++)
    for (int64_t c = 0; c < cols; c++) {
      const int64_t elem_offset = 2 + r*cols + c;
      result[elem_offset] = 0.0;
      for (int64_t k = 0; k < MX_COLS(first); k++)
        result[elem_offset] += first[2 + r*MX_COLS(first) + k] * second[2 + k*MX_COLS(second) + c];
      result[elem_offset] *= alpha;
    }
}

void
matrix_dot_and_add(
  const float alpha, const Matrix first, const Matrix second, const Matrix third, Matrix result
) {
  const int64_t rows = MX_ROWS(first);
  const int64_t cols = MX_COLS(second);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t r = 0; r < rows; r++)
    for (int64_t c = 0; c < cols; c++) {
      const int64_t elem_offset = 2 + r*cols + c;
      result[elem_offset] = third[elem_offset];
      for (int64_t k = 0; k < MX_COLS(first); k++)
        result[elem_offset] += first[2 + r*MX_COLS(first) + k] * second[2 + k*MX_COLS(second) + c];
      result[elem_offset] *= alpha;
    }
}

void
matrix_dot_and_apply(
  const float alpha, const Matrix first, const Matrix second, const char *function_name, Matrix result
) {
  const math_func_ptr_t func = math_func_from_name(function_name);

  const int64_t rows = MX_ROWS(first);
  const int64_t cols = MX_COLS(second);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t r = 0; r < rows; r++)
    for (int64_t c = 0; c < cols; c++) {
      const int64_t elem_offset = 2 + r*cols + c;
      result[elem_offset] = 0.0;
      for (int64_t k = 0; k < MX_COLS(first); k++)
        result[elem_offset] += first[2 + r*MX_COLS(first) + k] * second[2 + k*MX_COLS(second) + c];
      result[elem_offset] = func(alpha * result[elem_offset]);
    }
}

void
matrix_dot_nt(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  const int64_t rows = MX_ROWS(first);
  const int64_t cols = MX_ROWS(second);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t r = 0; r < rows; r++)
    for (int64_t c = 0; c < cols; c++) {
      const int64_t elem_offset = 2 + r*cols + c;
      result[elem_offset] = 0.0;
      for (int64_t k = 0; k < MX_COLS(first); k++)
        result[elem_offset] += first[2 + r*MX_COLS(first) + k] * second[2 + c*MX_COLS(second) + k];
      result[elem_offset] *= alpha;
    }
}

void
matrix_dot_tn(const float alpha, const Matrix first, const Matrix second, Matrix result) {
  const int64_t rows = MX_COLS(first);
  const int64_t cols = MX_COLS(second);

  MX_SET_ROWS(result, rows);
  MX_SET_COLS(result, cols);

  for (int64_t r = 0; r < rows; r++)
    for (int64_t c = 0; c < cols; c++) {
      const int64_t elem_offset = 2 + r*cols + c;
      result[elem_offset] = 0.0;
      for (int64_t k = 0; k < MX_ROWS(first); k++)
        result[elem_offset] += first[2 + r + k*rows]*second[2 + c + k*cols];
      result[elem_offset] *= alpha;
    }
}


#endif
