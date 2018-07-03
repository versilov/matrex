defmodule ArithmeticTest do
  use ExUnit.Case, async: true
  import Matrex

  [:byte, :int16, :int32, :int64, :float32, :float64]
  |> Enum.each(fn type ->
    test "#add adds two matrices of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2, 1], [3, 4, 6]], unquote(type))
      expected = new([[6, 4, 4], [7, 9, 12]], unquote(type))

      assert add(first, second) == expected
    end

    test "#add raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2], [3, 4]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
        add(first, second)
      end
    end

    test "#add adds scalar to each element of a matrix of type #{type}" do
      matrix = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      scalar = 3
      expected = new([[4, 5, 6], [7, 8, 9]], unquote(type))

      assert add(matrix, scalar) == expected
      assert add(scalar, matrix) == expected
    end

    test "#add/4 scales both operands before adding of type #{type}" do
      a = new("1 2 3; 4 5 6", unquote(type))
      b = new("3 2 1; 6 5 4", unquote(type))
      alpha = 2
      beta = 3

      expected = new("11 10 9; 26 25 24", unquote(type))

      assert add(a, b, alpha, beta) == expected
    end

    test "#apply/2 applies a function/1 on each element of the matrix of type #{type}" do
      function = &(&1 + 1)
      input = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      expected = new([[2, 3, 4], [5, 6, 7]], unquote(type))

      assert Matrex.apply(input, function) == expected
    end

    test "#apply/2 applies a function/3 on each element of the matrix of type #{type}" do
      function = fn element, {row_index, column_index} ->
        element + row_index + column_index
      end

      input = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      expected = new([[3, 5, 7], [7, 9, 11]], unquote(type))

      assert Matrex.apply(input, function) == expected
    end

    test "#apply/3 applies a function on each element of the matrices of type #{type}" do
      function = &(&1 + &2)
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[2, 3, 4], [5, 6, 7]], unquote(type))
      expected = new([[3, 5, 7], [9, 11, 13]], unquote(type))

      assert Matrex.apply(first, second, function) == expected
    end

    test "#divide divides two matrices of type #{type}" do
      first = new([[1, 2, 6], [9, 10, 18]], unquote(type))
      second = new([[2, 2, 3], [3, 5, 6]], unquote(type))
      expected = new([[0.5, 1, 2], [3, 2, 3]], unquote(type))

      assert divide(first, second) == expected
    end

    test "#divide divides matrix by scalar of type #{type}" do
      dividend = new([[10, 20, 25], [8, 9, 4]], unquote(type))
      expected = new("5.0 10.0 12.5; 4.0 4.5 2.0", unquote(type))

      assert divide(dividend, 2) == expected
    end

    test "#divide divides scalar by matrix of type #{type}" do
      matrix = new([[10, 20, 25], [8, 16, 4]], unquote(type))
      expected = new("10.0 5.0 4.0; 12.5 6.25 25.0", unquote(type))

      assert divide(100, matrix) == expected
    end

    test "#divide raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2], [3, 4]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
        divide(first, second)
      end
    end

    test "#dot multiplies two matrices of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[1, 2], [3, 4], [5, 6]], unquote(type))
      expected = new([[22, 28], [49, 64]], unquote(type))

      assert dot(first, second) == expected
    end

    test "#dot raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[2, 2, 3], [3, 5, 6]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
        dot(first, second)
      end
    end

    test "#dot_and_add multiplies two matrices and adds the third of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[1, 2], [3, 4], [5, 6]], unquote(type))
      third = new([[1, 2], [3, 4]], unquote(type))
      expected = new([[23, 30], [52, 68]], unquote(type))

      assert dot_and_add(first, second, third) == expected
    end

    test "#dot_and_add raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[1, 2], [3, 4], [5, 6]], unquote(type))
      third = new([[1, 2], [3, 4], [5, 6]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
        dot_and_add(first, second, third)
      end
    end

    test "#dot_nt multiplies two matrices, second needing to be transposed of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[1, 3, 5], [2, 4, 6]], unquote(type))
      expected = new([[22, 28], [49, 64]], unquote(type))

      assert dot_nt(first, second) == expected
    end

    test "#dot_nt raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[2, 2, 3, 5], [3, 5, 6, 7]], unquote(type))

      assert_raise FunctionClauseError, fn ->
        dot_nt(first, second)
      end
    end

    test "#dot_tn multiplies two matrices, first needing to be transposed of type #{type}" do
      first = new([[1, 4], [2, 5], [3, 6]], unquote(type))
      second = new([[1, 2], [3, 4], [5, 6]], unquote(type))
      expected = new([[22, 28], [49, 64]], unquote(type))

      assert dot_tn(first, second) == expected
    end

    test "#dot_tn raises when sizes do not match of type #{type}" do
      first = new([[1, 4], [2, 5], [3, 6]], unquote(type))
      second = new([[2, 2, 3, 5], [3, 5, 6, 7]], unquote(type))

      assert_raise FunctionClauseError, fn ->
        dot_tn(first, second)
      end
    end

    test "#multiply performs elementwise multiplication of two matrices of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2, 1], [3, 4, 6]], unquote(type))
      expected = new([[5, 4, 3], [12, 20, 36]], unquote(type))

      assert multiply(first, second) == expected
    end

    test "#multiply raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2], [3, 4]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
        multiply(first, second)
      end
    end

    test "#multiply multiplies matrix element by a scalar of type #{type}" do
      matrix = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      scalar = 2
      expected = new([[2, 4, 6], [8, 10, 12]], unquote(type))

      assert multiply(matrix, scalar) == expected
    end

    test "#neg negates matrix of type #{type}" do
      matrix = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      expected = new([[-1, -2, -3], [-4, -5, -6]], unquote(type))

      assert neg(matrix) == expected
    end

    test "#square returns square of a matrix of type #{type}" do
      matrix = reshape(1..10, {2, 5}, unquote(type))
      expected = new([[1, 4, 9, 16, 25], [36, 49, 64, 81, 100]], unquote(type))

      assert square(matrix) == expected
    end

    test "#subtract subtracts two matrices of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2, 1], [3, 4, 6]], unquote(type))
      expected = new([[-4, 0, 2], [1, 1, 0]], unquote(type))

      assert subtract(first, second) == expected
    end

    test "#subtract raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2], [3, 4]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch./, fn ->
        subtract(first, second)
      end
    end

    test "#subtract subtracts matrix from scalar of type #{type}" do
      input = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      expected = new([[0, -1, -2], [-3, -4, -5]], unquote(type))

      assert subtract(1, input) == expected
    end

    test "#subtract subtracts scalar from matrix of type #{type}" do
      input = reshape(1..6, {2, 3}, unquote(type))
      expected = reshape(0..5, {2, 3}, unquote(type))

      assert subtract(input, 1) == expected
    end

    test "#subtract_inverse subtracts the second matrix from the first of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[5, 2, 1], [3, 4, 6]], unquote(type))
      expected = new([[4, 0, -2], [-1, -1, 0]], unquote(type))

      assert subtract_inverse(first, second) == expected
    end

    test "#subtract_inverse subtracts matrix from scalar of type #{type}" do
      input = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      expected = new([[0, -1, -2], [-3, -4, -5]], unquote(type))

      assert subtract_inverse(input, 1) == expected
    end

    test "#subtract_inverse subtracts scalar from matrix of type #{type}" do
      input = reshape(1..6, {2, 3}, unquote(type))
      expected = reshape(0..5, {2, 3}, unquote(type))

      assert subtract_inverse(1, input) == expected
    end
  end)

  [:float32, :float64]
  |> Enum.each(fn type ->
    test "#apply/2 applies a math function on each element of the matrix of type #{type}" do
      input = new([[4, 16, 9], [25, 49, 36]], unquote(type))
      expected = new([[2, 4, 3], [5, 7, 6]], unquote(type))

      assert Matrex.apply(input, :sqrt) == expected
    end

    test "#apply/2 applies a math function on each element of the matrix larger, than 100_000 elements of type #{
           type
         }" do
      input = random({500, 500}, unquote(type))
      output = Matrex.apply(input, :exp)

      assert Float.round(at(output, 325, 414), 5) ==
               Float.round(:math.exp(at(input, 325, 414)), 5)
    end

    test "#divide divides matrix by zero of type #{type}" do
      dividend = new([[10, 20, 25], [8, 9, 4]], unquote(type))
      expected = new("Inf Inf Inf; Inf Inf Inf", unquote(type))

      assert divide(dividend, 0) == expected
    end

    test "#dot_and_apply raises when sizes do not match of type #{type}" do
      first = new([[1, 2, 3], [4, 5, 6]], unquote(type))
      second = new([[2, 2, 3], [3, 5, 6]], unquote(type))

      assert_raise ArgumentError, ~r/matrices' shapes mismatch/, fn ->
        dot_and_apply(first, second, :sigmoid)
      end
    end
  end)

  test "#dot_and_apply multiplies two matrices and applies function to the result of type :float32" do
    first = new([[1, 2, 3], [4, 5, 6]])
    second = new([[1, 2], [3, 4], [5, 6]])

    expected =
      new([
        [-0.008851309306919575, 0.2709057927131653],
        [-0.9537526369094849, 0.9200260639190674]
      ])

    assert dot_and_apply(first, second, :sin) == expected
  end

  test "#dot_and_apply multiplies two matrices and applies function to the result of type :float64" do
    first = new([[1, 2, 3], [4, 5, 6]], :float64)
    second = new([[1, 2], [3, 4], [5, 6]], :float64)

    expected =
      new(
        [
          [-0.008851309290403876, 0.27090578830786904],
          [-0.9537526527594719, 0.9200260381967907]
        ],
        :float64
      )

    assert dot_and_apply(first, second, :sin) == expected
  end
end
