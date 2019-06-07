#include "../include/matrix.h"
#include "../include/matrix_linalg.h"

/*

S_ik = \sum_{j=1}^{k-1} ( l_ij * l_kj )

if i == k do
  l_kk = \sqrt{ a_kk - S_ik } 
else
  l_ik = \frac{1}{l_kk} \( a_ik - S_ik \)
end

*/

void
matrix_cholesky(const Matrix matrix, Matrix result) {
  size_t N = MX_ROWS(matrix);
  size_t cols = MX_COLS(matrix);

  for (size_t i = 0; i < N*cols; i++)
    result[2 + i] = 0.0;

  MX_SET_ROWS(result, N);
  MX_SET_COLS(result, N);

  for (size_t i = 0; i < N; i++)
      for (size_t k = 0; k < (i+1); k++) {

          float ts = 0.0;
          for (size_t j = 0; j < k; j++)
            ts += result[2 + i*cols + j] * result[2 + k*cols + j];

          result[2 + i*cols + k] = (i == k) ?
                          sqrt(fmax(matrix[2 + i*cols + i] - ts, 0.0)) :
                          (1.0 / result[2 + k*cols + k]) * (matrix[2 + i*cols + k] - ts);
      }

}

/*
N = size(L)[1]
v = zeros(N);

for i in 1:N
    v[i] = ( cK[i] - sum(v .* L[i, :]) ) ./ L[i, i]
end
*/

void
matrix_solve(const Matrix matrix, const Matrix beta, Matrix result) {
  const size_t N = MX_ROWS(matrix);
  const size_t cols = MX_COLS(matrix);

  MX_SET_ROWS(result, N);
  MX_SET_COLS(result, 1);

  for (size_t i = 0; i < N; i++)
    result[2 + i] = 0.0;

  for (size_t r = 0; r < N; r++) {
    const int64_t elem_offset = 2 + r*cols;
    // sum( v .* L[r,:] )
    float row_sum = 0.0;
    for (size_t c = 0; c < r; c++)
      row_sum += result[2 + c] * matrix[elem_offset + c];
    
    result[2 + r] = (beta[2 + r] - row_sum) / matrix[elem_offset + r];
  }

}
