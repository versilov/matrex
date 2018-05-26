#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "c/test_util.c"
#include "c/matrix_test.c"

int main() {
  // Tests for: c/matrix_test.c
  test_matrix_clone();
  test_matrix_free();
  test_matrix_new();
  test_matrix_fill();
  test_matrix_equal();
  test_matrix_add();
  test_matrix_argmax();
  test_matrix_divide();
  test_matrix_dot();
  test_matrix_dot_and_add();
  test_matrix_dot_nt();
  test_matrix_dot_tn();
  test_matrix_first();
  test_matrix_inspect();
  test_matrix_inspect_internal();
  test_matrix_max();
  test_matrix_multiply();
  test_matrix_multiply_with_scalar();
  test_matrix_subtract();
  test_matrix_sum();
  test_matrix_transpose();
}
