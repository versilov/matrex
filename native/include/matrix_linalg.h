#ifndef INCLUDED_MATRIX_INV_H
#define INCLUDED_MATRIX_INV_H

#include "matrix.h"

void
matrix_cholesky(const Matrix matrix, Matrix result);

void
matrix_solve(const Matrix matrix, const Matrix b, Matrix result);

#endif
