#ifndef INCLUDED_MATRIX_H
#define INCLUDED_MATRIX_H

#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "./utils.h"

typedef float* Matrix;

#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)
#define MX_BYTE_SIZE(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)*4
#define MX_DATA_BYTE_SIZE(matrix) (((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1])*4

void
matrix_clone(Matrix destination, Matrix source);

void
matrix_free(Matrix *matrix);

Matrix
matrix_new(uint32_t rows, uint32_t columns);

void
matrix_fill(Matrix matrix, int32_t value);

void
matrix_random(Matrix matrix);

void
matrix_zeros(Matrix matrix);

void
matrix_eye(Matrix matrix, int32_t value);

int32_t
matrix_equal(Matrix first, Matrix second);

void
matrix_add(const Matrix first, const Matrix second, Matrix result);

int
matrix_apply(const Matrix matrix, char* function_name, Matrix result);

int32_t
matrix_argmax(const Matrix matrix);

void
matrix_divide(const Matrix first, const Matrix second, Matrix result);

void
matrix_dot(const Matrix first, const Matrix second, Matrix result);

void
matrix_dot_and_add(
  const Matrix first, const Matrix second, const Matrix third, Matrix result
);

void
matrix_dot_nt(const Matrix first, const Matrix second, Matrix result);

void
matrix_dot_tn(const Matrix first, const Matrix second, Matrix result);

float
matrix_first(const Matrix matrix);

void
matrix_inspect(const Matrix matrix);

void
matrix_inspect_internal(const Matrix matrix, int32_t indentation);

float
matrix_max(const Matrix matrix);

void
matrix_multiply(const Matrix first, const Matrix second, Matrix result);

void
matrix_multiply_with_scalar(
  const Matrix matrix, const float scalar, Matrix result
);

void
matrix_substract(const Matrix first, const Matrix second, Matrix result);

double
matrix_sum(const Matrix matrix);

void
matrix_transpose(const Matrix matrix, Matrix result);
#endif
