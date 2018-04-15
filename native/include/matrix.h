#ifndef INCLUDED_MATRIX_H
#define INCLUDED_MATRIX_H

#include <cblas.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "./utils.h"

typedef float* Matrix;

void
matrix_clone(Matrix destination, Matrix source);

void
matrix_free(Matrix *matrix);

Matrix
matrix_new(int32_t rows, int32_t columns);

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
