#include "../include/matrix.h"



void
matrix_decomp_chol(const Matrix matrix, Matrix result) {
  size_t data_size = MX_BYTE_SIZE(matrix);

  memcpy(result, matrix, data_size);
  size_t N = MX_ROWS(matrix);

  for (size_t r = 0; r < N; r++)
      for (size_t c = 0; c < (r+1); c++) {
          float ts = 0;
          for (size_t k = 0; k < c; k++)
            ts += result[2 + r*N + k] * result[2 + c*N + k];

          float v = (r == c) ?
                          sqrt(result[r * N + r] - ts) :
                          (1.0 / result[c * N + c] * (result[2 + r*N + c] - ts));

          result[2 + r*N + c] = v;
      }

}

/*
N = size(L)[1]
v = zeros(N);

for i in 1:N
    v[i] = ( cK[i] - sum(v .* L[i, :]) ) ./ L[i, i]
    @show v
end
*/

void
matrix_solve(const Matrix matrix, const Matrix beta, Matrix result) {
  const size_t N = MX_ROWS(matrix);

  MX_SET_ROWS(result, N);
  MX_SET_COLS(result, 1);

  for (size_t i = 0; i < N; i++)
    result[2 + i] = 0.0;

  for (size_t r = 0; r < N; r++) {
    // sum(v .* L[i, :])
    float ts = 0;
    for (size_t c = 0; c < N; c++)
      ts += result[2 + r*N + c] * matrix[2 + r*N + c];
    
    result[2 + r] = (beta[2 + r] - ts) / matrix[2 + r];
  }

}
