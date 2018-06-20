defmodule ArithmeticTest do
  use ExUnit.Case, async: true
  import Matrex

  test "#add adds two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[6, 4, 4], [7, 9, 12]])

    assert Matrex.add(first, second) == expected
  end

  test "#add raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
      Matrex.add(first, second)
    end
  end

  test "#add adds scalar to each element of a matrix" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    scalar = 3
    expected = Matrex.new([[4, 5, 6], [7, 8, 9]])

    assert Matrex.add(matrix, scalar) == expected
    assert Matrex.add(scalar, matrix) == expected
  end

  test "#add/4 scales both operands before adding" do
    a = Matrex.new("1 2 3; 4 5 6")
    b = Matrex.new("3 2 1; 6 5 4")
    alpha = 2
    beta = 3

    expected = Matrex.new("11 10 9; 26 25 24")

    assert Matrex.add(a, b, alpha, beta) == expected
  end

  test "#apply/2 applies a math function on each element of the matrix" do
    input = Matrex.new([[4, 16, 9], [25, 49, 36]])
    expected = Matrex.new([[2, 4, 3], [5, 7, 6]])

    assert Matrex.apply(input, :sqrt) == expected
  end

  test "#apply/2 applies a math function on each element of the matrix larger, than 100_000 elements" do
    input = Matrex.random(500)
    output = Matrex.apply(input, :exp)

    assert Float.round(Matrex.at(output, 325, 414), 5) ==
             Float.round(:math.exp(Matrex.at(input, 325, 414)), 5)
  end

  test "#apply/2 applies a function/1 on each element of the matrix" do
    function = &(&1 + 1)
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[2, 3, 4], [5, 6, 7]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/2 applies a function/3 on each element of the matrix" do
    function = fn element, {row_index, column_index} ->
      element + row_index + column_index
    end

    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[3, 5, 7], [7, 9, 11]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/3 applies a function on each element of the matrices" do
    function = &(&1 + &2)
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[2, 3, 4], [5, 6, 7]])
    expected = Matrex.new([[3, 5, 7], [9, 11, 13]])

    assert Matrex.apply(first, second, function) == expected
  end

  test "#divide divides two matrices" do
    first = Matrex.new([[1, 2, 6], [9, 10, 18]])
    second = Matrex.new([[2, 2, 3], [3, 5, 6]])
    expected = Matrex.new([[0.5, 1, 2], [3, 2, 3]])

    assert Matrex.divide(first, second) == expected
  end

  test "#divide divides matrix by scalar" do
    dividend = Matrex.new([[10, 20, 25], [8, 9, 4]])
    expected = Matrex.new("5.0 10.0 12.5; 4.0 4.5 2.0")

    assert Matrex.divide(dividend, 2) == expected
  end

  test "#divide divides matrix by zero" do
    dividend = Matrex.new([[10, 20, 25], [8, 9, 4]])
    expected = Matrex.new("Inf Inf Inf; Inf Inf Inf")

    assert Matrex.divide(dividend, 0) == expected
  end

  test "#divide divides scalar by matrix" do
    matrix = Matrex.new([[10, 20, 25], [8, 16, 4]])
    expected = Matrex.new("10.0 5.0 4.0; 12.5 6.25 25.0")

    assert Matrex.divide(100, matrix) == expected
  end

  test "#divide raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
      Matrex.divide(first, second)
    end
  end

  test "#dot multiplies two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[1, 2], [3, 4], [5, 6]])
    expected = Matrex.new([[22, 28], [49, 64]])

    assert Matrex.dot(first, second) == expected
  end

  test "#dot raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[2, 2, 3], [3, 5, 6]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
      Matrex.dot(first, second)
    end
  end

  test "#dot_and_add multiplies two matrices and adds the third" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[1, 2], [3, 4], [5, 6]])
    third = Matrex.new([[1, 2], [3, 4]])
    expected = Matrex.new([[23, 30], [52, 68]])

    assert Matrex.dot_and_add(first, second, third) == expected
  end

  test "#dot_and_add raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[1, 2], [3, 4], [5, 6]])
    third = Matrex.new([[1, 2], [3, 4], [5, 6]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
      Matrex.dot_and_add(first, second, third)
    end
  end

  test "#dot_and_apply multiplies two matrices and applies function to the result" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[1, 2], [3, 4], [5, 6]])

    expected =
      Matrex.new([
        [-0.008851309306919575, 0.2709057927131653],
        [-0.9537526369094849, 0.9200260639190674]
      ])

    assert Matrex.dot_and_apply(first, second, :sin) == expected
  end

  test "#dot_and_apply raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[2, 2, 3], [3, 5, 6]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
      Matrex.dot_and_apply(first, second, :sigmoid)
    end
  end

  test "#dot_nt multiplies two matrices, second needing to be transposed" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[1, 3, 5], [2, 4, 6]])
    expected = Matrex.new([[22, 28], [49, 64]])

    assert Matrex.dot_nt(first, second) == expected
  end

  test "#dot_nt raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[2, 2, 3, 5], [3, 5, 6, 7]])

    assert_raise FunctionClauseError, fn ->
      Matrex.dot_nt(first, second)
    end
  end

  test "#dot_tn multiplies two matrices, first needing to be transposed" do
    first = Matrex.new([[1, 4], [2, 5], [3, 6]])
    second = Matrex.new([[1, 2], [3, 4], [5, 6]])
    expected = Matrex.new([[22, 28], [49, 64]])

    assert Matrex.dot_tn(first, second) == expected
  end

  test "#dot_tn raises when sizes do not match" do
    first = Matrex.new([[1, 4], [2, 5], [3, 6]])
    second = Matrex.new([[2, 2, 3, 5], [3, 5, 6, 7]])

    assert_raise FunctionClauseError, fn ->
      Matrex.dot_tn(first, second)
    end
  end

  test "#multiply performs elementwise multiplication of two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[5, 4, 3], [12, 20, 36]])

    assert Matrex.multiply(first, second) == expected
  end

  test "#multiply raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
      Matrex.multiply(first, second)
    end
  end

  test "#multiply multiplies matrix element by a scalar" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    scalar = 2
    expected = Matrex.new([[2, 4, 6], [8, 10, 12]])

    assert Matrex.multiply(matrix, scalar) == expected
  end

  test "#neg negates matrix" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[-1, -2, -3], [-4, -5, -6]])

    assert Matrex.neg(matrix) == expected
  end

  test "#subtract subtracts two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[-4, 0, 2], [1, 1, 0]])

    assert Matrex.subtract(first, second) == expected
  end

  test "#subtract raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
      Matrex.subtract(first, second)
    end
  end

  test "#subtract subtracts matrix from scalar" do
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[0, -1, -2], [-3, -4, -5]])

    assert Matrex.subtract(1, input) == expected
  end

  test "#subtract subtracts scalar from matrix" do
    input = Matrex.reshape(1..6, {2, 3})
    expected = Matrex.reshape(0..5, {2, 3})

    assert Matrex.subtract(input, 1) == expected
  end

  test "#subtract_inverse subtracts the second matrix from the first" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[4, 0, -2], [-1, -1, 0]])

    assert Matrex.subtract_inverse(first, second) == expected
  end

  test "#subtract_inverse subtracts matrix from scalar" do
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[0, -1, -2], [-3, -4, -5]])

    assert Matrex.subtract_inverse(input, 1) == expected
  end

  test "#subtract_inverse subtracts scalar from matrix" do
    input = Matrex.reshape(1..6, {2, 3})
    expected = Matrex.reshape(0..5, {2, 3})

    assert Matrex.subtract_inverse(1, input) == expected
  end
end
