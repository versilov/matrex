#ifndef INCLUDED_MATRIX_H
#define INCLUDED_MATRIX_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "./utils.h"

typedef float* Matrix;

#define MX_ELEMENT_SIZE 4 // Size of float
#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)
#define MX_BYTE_SIZE(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)*MX_ELEMENT_SIZE
#define MX_DATA_BYTE_SIZE(matrix) (((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1])*MX_ELEMENT_SIZE


void
matrix_dot_pure(const Matrix first, const Matrix second, Matrix result);


void
matrix_clone(Matrix destination, Matrix source);

void
matrix_free(Matrix *matrix);

Matrix
matrix_new(uint32_t rows, uint32_t columns);

int32_t
matrix_equal(Matrix first, Matrix second);

void
matrix_add(const Matrix first, const Matrix second, const float alpha, const float beta, Matrix result);

void
matrix_add_scalar(const Matrix first, const float scalar, Matrix result);

int
matrix_apply(const Matrix matrix, char* function_name, Matrix result);

typedef float (*math_func_ptr_t)(float);

math_func_ptr_t math_func_from_name(const char* name);

int32_t
matrix_argmax(const Matrix matrix);

void
matrix_concat_columns(const Matrix first, const Matrix second, Matrix result);

void
matrix_divide(const Matrix first, const Matrix second, Matrix result);

void
matrix_divide_scalar(const float scalar, const Matrix divisor, Matrix result);

void
matrix_divide_by_scalar(const Matrix dividend, const float scalar, Matrix result);

void
matrix_eye(Matrix matrix, const float value);

void
matrix_diagonal(const Matrix matrix, const uint64_t diag_size, Matrix result);

void
matrix_fill(Matrix matrix, const float value);

int32_t
matrix_find(const Matrix matrix, const float value);

int32_t
matrix_find_nan(const Matrix matrix);

float
matrix_first(const Matrix matrix);

void
matrix_from_range(const int64_t from, const int64_t to, const int64_t rows, const int64_t cols, Matrix result);

void
matrix_inspect(const Matrix matrix);

void
matrix_inspect_internal(const Matrix matrix, int32_t indentation);

float
matrix_max(const Matrix matrix);

float
matrix_min(const Matrix matrix);

float
matrix_max_finite(const Matrix matrix);

float
matrix_min_finite(const Matrix matrix);

void
matrix_multiply(const Matrix first, const Matrix second, Matrix result);

void
matrix_multiply_with_scalar(const Matrix matrix, const float scalar, Matrix result);

void
matrix_neg(const Matrix matrix, Matrix result);

void
matrix_normalize(const Matrix matrix, Matrix result);

void
matrix_random(Matrix matrix);

void
matrix_resize(const Matrix matrix, const int32_t new_rows, const int32_t new_cols, Matrix result);

void
matrix_set(const Matrix matrix, const uint32_t row, const uint32_t column, const float scalar, Matrix result);

void
matrix_set_column(const Matrix matrix, const uint32_t column, const Matrix column_matrix, Matrix result);

void
matrix_submatrix(const Matrix matrix, const uint32_t row_from, const uint32_t row_to,
  const uint32_t column_from, const uint32_t column_to, Matrix result);

void
matrix_subtract(const Matrix first, const Matrix second, Matrix result);

void
matrix_subtract_from_scalar(const float scalar, const Matrix matrix, Matrix result);

double
matrix_sum(const Matrix matrix);

void
matrix_transpose(const Matrix matrix, Matrix result);

void
matrix_zeros(Matrix matrix);

#endif
