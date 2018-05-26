#include "../../native/include/matrix.h"

static void test_matrix_clone() {
  Matrix source      = matrix_new(1, 3);
  Matrix destination = matrix_new(1, 3);

  source[2] = 1;
  source[3] = 2;
  source[4] = 3;

  destination[2] = 10;
  destination[3] = 20;
  destination[4] = 30;

  matrix_clone(destination, source);

  for(int32_t index = 0; index < 5; index += 1) {
    assert(destination[index] == source[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_free() {
  Matrix matrix = matrix_new(1, 2);

  matrix_free(&matrix);

  assert(matrix == NULL); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_new() {
  Matrix matrix = matrix_new(1, 2);

  assert(MX_ROWS(matrix) == 1); /* LCOV_EXCL_BR_LINE */
  assert(MX_COLS(matrix) == 2); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_fill() {
  Matrix matrix = matrix_new(1, 2);

  matrix_fill(matrix, 3);

  assert(MX_ROWS(matrix) == 1); /* LCOV_EXCL_BR_LINE */
  assert(MX_COLS(matrix) == 2); /* LCOV_EXCL_BR_LINE */
  assert(matrix[2] == 3); /* LCOV_EXCL_BR_LINE */
  assert(matrix[3] == 3); /* LCOV_EXCL_BR_LINE */

}

static void test_matrix_equal() {
  float first[8]  = {2, 3, 1, 2, 3, 4, 5, 6 };
  float second[8] = {2, 3, 1, 2, 3, 4, 5, 6 };
  float third[8]  = {2, 3, 5, 2, 1, 3, 4, 6 };
  float fourth[8] = {3, 2, 5, 2, 1, 3, 4, 6 };
  float fifth[10] = {2, 4, 5, 2, 1, 3, 4, 6, 7, 8};

  *(uint32_t*)&first[0] = 2;
  *(uint32_t*)&first[1] = 3;
  *(uint32_t*)&second[0] = 2;
  *(uint32_t*)&second[1] = 3;
  *(uint32_t*)&third[0] = 2;
  *(uint32_t*)&third[1] = 3;

  assert(matrix_equal(first, second)); /* LCOV_EXCL_BR_LINE */

  assert(matrix_equal(first, second) == 1); /* LCOV_EXCL_BR_LINE */
  assert(matrix_equal(first, third)  == 0); /* LCOV_EXCL_BR_LINE */
  assert(matrix_equal(first, fourth) == 0); /* LCOV_EXCL_BR_LINE */
  assert(matrix_equal(first, fifth)  == 0); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_add() {
  float first[8]    = {2, 3, 1, 2, 3, 4, 5, 6 };
  float second[8]   = {2, 3, 5, 2, 1, 3, 4, 6 };
  float expected[8] = {2, 3, 6, 4, 4, 7, 9, 12};
  float result[8];

  matrix_add(first, second, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_argmax() {
  float first[8]  = {2, 3, 1, 2, 3, 4, 5, 6};
  float second[8] = {2, 3, 8, 3, 4, 5, 6, 7};
  float third[8]  = {2, 3, 8, 3, 4, 9, 6, 7};

  assert(matrix_argmax(first)  == 5); /* LCOV_EXCL_BR_LINE */
  assert(matrix_argmax(second) == 0); /* LCOV_EXCL_BR_LINE */
  assert(matrix_argmax(third)  == 3); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_divide() {
  float first[8]    = {2, 3, 1,   2, 6, 9, 10, 18};
  float second[8]   = {2, 3, 2,   2, 3, 3, 5,  6 };
  float expected[8] = {2, 3, 0.5, 1, 2, 3, 2,  3 };
  float result[8];

  matrix_divide(first, second, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_dot() {
  float first[8]    = {2, 3, 1, 2, 3, 4, 5, 6};
  float second[8]   = {3, 2, 1, 2, 3, 4, 5, 6};
  float expected[8] = {2, 2, 22, 28, 49, 64};
  float result[8];

  matrix_dot(first, second, result);

  for(int32_t index = 0; index < 6; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_dot_and_add() {
  float first[8]    = {2, 3, 1, 2, 3, 4, 5, 6};
  float second[8]   = {3, 2, 1, 2, 3, 4, 5, 6};
  float third[8]    = {2, 2, 1, 2, 3, 4};
  float expected[8] = {2, 2, 23, 30, 52, 68};
  float result[8];

  matrix_dot_and_add(first, second, third, result);

  for(int32_t index = 0; index < 6; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_dot_nt() {
  float first[8]    = {2, 3, 1, 2, 3, 4, 5, 6};
  float second[8]   = {2, 3, 1, 3, 5, 2, 4, 6};
  float expected[8] = {2, 2, 22, 28, 49, 64};
  float result[8];

  matrix_dot_nt(first, second, result);

  for(int32_t index = 0; index < 6; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_dot_tn() {
  float first[8]    = {3, 2, 1, 4, 2, 5, 3, 6};
  float second[8]   = {3, 2, 1, 2, 3, 4, 5, 6};
  float expected[8] = {2, 2, 22, 28, 49, 64};
  float result[8];

  matrix_dot_tn(first, second, result);

  for(int32_t index = 0; index < 6; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_first() {
  float matrix[8] = {2, 3, 1, 4, 2, 5, 3, 6};

  assert(matrix_first(matrix) == 1); /* LCOV_EXCL_BR_LINE */
}

static void
test_matrix_inspect_body() {
  Matrix matrix = matrix_new(2, 3);

  for (int32_t index = 2; index < 8; index += 1) {
    matrix[index] = index - 2;
  }

  matrix_inspect(matrix);
}

static void test_matrix_inspect() {
  char *result   = capture_stdout(test_matrix_inspect_body);
  char *expected =
    "<#Matrix\n"
    "  rows:    2.000000\n"
    "  columns: 3.000000\n"
    "  values:  0.000000 1.000000 2.000000 3.000000 4.000000 5.000000>\n";

  int32_t result_length   = strlen(result  );
  int32_t expected_length = strlen(expected);

  assert(result_length == expected_length); /* LCOV_EXCL_BR_LINE */

  for(int32_t index = 0; index <= result_length; index += 1) {
    assert(result[index] == expected[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void
test_matrix_inspect_internal_body() {
  Matrix matrix = matrix_new(2, 3);

  for (int32_t index = 2; index < 8; index += 1) {
    matrix[index] = index - 2;
  }

  matrix_inspect_internal(matrix, 3);
}

static void test_matrix_inspect_internal() {
  char *result   = capture_stdout(test_matrix_inspect_internal_body);
  char *expected =
    "<#Matrix\n"
    "     rows:    2.000000\n"
    "     columns: 3.000000\n"
    "     values:  0.000000 1.000000 2.000000 3.000000 4.000000 5.000000>";

  int32_t result_length   = strlen(result  );
  int32_t expected_length = strlen(expected);

  assert(result_length == expected_length); /* LCOV_EXCL_BR_LINE */

  for(int32_t index = 0; index <= result_length; index += 1) {
    assert(result[index] == expected[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_max() {
  float matrix[8] = {2, 3, 1, 4, 2, 5, 3, 6};

  assert(matrix_max(matrix) == 6); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_multiply() {
  float first[8]    = {2, 3, 1, 2, 3, 4,  5,  6 };
  float second[8]   = {2, 3, 5, 2, 1, 3,  4,  6 };
  float expected[8] = {2, 3, 5, 4, 3, 12, 20, 36};
  float result[8];

  matrix_multiply(first, second, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_multiply_with_scalar() {
  float matrix[8]   = {2, 3, 1, 2, 3, 4, 5, 6};
  float scalar      = 2;
  float expected[8] = {2, 3, 2, 4, 6, 8, 10, 12};
  float result[8];

  matrix_multiply_with_scalar(matrix, scalar, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_subtract() {
  float first[8]    = {2, 3,  1, 2, 3, 4, 5, 6};
  float second[8]   = {2, 3,  5, 2, 1, 3, 4, 6};
  float expected[8] = {2, 3, -4, 0, 2, 1, 1, 0};
  float result[8];

  matrix_subtract(first, second, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}

static void test_matrix_sum() {
  float matrix[8] = {2, 3, 1, 4, 2, 5, 3, 6};

  assert(matrix_sum(matrix) == 21); /* LCOV_EXCL_BR_LINE */
}

static void test_matrix_transpose() {
  float matrix[8]   = {2, 3, 1, 2, 3, 4, 5, 6};
  float expected[8] = {3, 2, 1, 4, 2, 5, 3, 6};
  float result[8];

  matrix_transpose(matrix, result);

  for(int32_t index = 0; index < 8; index += 1) {
    assert(expected[index] == result[index]); /* LCOV_EXCL_BR_LINE */
  }
}
