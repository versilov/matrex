#ifndef INCLUDED_MATRIX_DOT_H
#define INCLUDED_MATRIX_DOT_H

#include "matrix.h"

void
matrix_dot(const float alpha, const Matrix first, const Matrix second, Matrix result);

void
matrix_dot_and_add(
  const float alpha, const Matrix first, const Matrix second, const Matrix third, Matrix result
);

void
matrix_dot_and_apply(
  const float alpha, const Matrix first, const Matrix second, const char *function_name, Matrix result
);

void
matrix_dot_nt(const float alpha, const Matrix first, const Matrix second, Matrix result);

void
matrix_dot_tn(const float alpha, const Matrix first, const Matrix second, Matrix result);

#endif
