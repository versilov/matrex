defmodule MatrexTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO

  test "#add adds two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[6, 4, 4], [7, 9, 12]])

    assert Matrex.add(first, second) == expected
  end

  test "#add raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
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

  test "#apply/2 applies a function/2 on each element of the matrix" do
    function = fn element, index -> element + index end
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[2, 4, 6], [8, 10, 12]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/2 applies a function/3 on each element of the matrix" do
    function = fn element, row_index, column_index ->
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

  test "#argmax returns the index of the maximal element" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[8, 3, 4], [5, 6, 7]])
    third = Matrex.new([[8, 3, 4], [9, 6, 7]])

    assert Matrex.argmax(first) == 6
    assert Matrex.argmax(second) == 1
    assert Matrex.argmax(third[2]) == 1
  end

  test "#at returns element at the given position" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])

    assert Matrex.at(matrix, 1, 3) == 3
  end

  test "#at raises when row position is out of range" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, 0, 3)
    end

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, -2, 1)
    end
  end

  test "#at raises when column position is out of range" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, 2, 5)
    end

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, 1, -1)
    end
  end

  test "#column returns column of the matrix" do
    matrix =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    assert Matrex.column(matrix, 4) == Matrex.new([[7], [13], [19], [25], [1]])
  end

  test "#column_as_list returns column of matrix" do
    matrix =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    assert Matrex.column_to_list(matrix, 3) == [5, 6, 12, 18, 24]
  end

  test "#divide divides two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 6], [9, 10, 18]])
    second = Matrex.new(2, 3, [[2, 2, 3], [3, 5, 6]])
    expected = Matrex.new(2, 3, [[0.5, 1, 2], [3, 2, 3]])

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
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 2, [[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch./, fn ->
      Matrex.divide(first, second)
    end
  end

  test "#dot multiplies two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(3, 2, [[1, 2], [3, 4], [5, 6]])
    expected = Matrex.new(2, 2, [[22, 28], [49, 64]])

    assert Matrex.dot(first, second) == expected
  end

  test "#dot raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[2, 2, 3], [3, 5, 6]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch./, fn ->
      Matrex.dot(first, second)
    end
  end

  test "#dot_and_add multiplies two matrices and adds the third" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(3, 2, [[1, 2], [3, 4], [5, 6]])
    third = Matrex.new(2, 2, [[1, 2], [3, 4]])
    expected = Matrex.new(2, 2, [[23, 30], [52, 68]])

    assert Matrex.dot_and_add(first, second, third) == expected
  end

  test "#dot_and_add raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(3, 2, [[1, 2], [3, 4], [5, 6]])
    third = Matrex.new(3, 2, [[1, 2], [3, 4], [5, 6]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
      Matrex.dot_and_add(first, second, third)
    end
  end

  test "#dot_nt multiplies two matrices, second needing to be transposed" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[1, 3, 5], [2, 4, 6]])
    expected = Matrex.new(2, 2, [[22, 28], [49, 64]])

    assert Matrex.dot_nt(first, second) == expected
  end

  test "#dot_nt raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 4, [[2, 2, 3, 5], [3, 5, 6, 7]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
      Matrex.dot_nt(first, second)
    end
  end

  test "#dot_tn multiplies two matrices, first needing to be transposed" do
    first = Matrex.new(3, 2, [[1, 4], [2, 5], [3, 6]])
    second = Matrex.new(3, 2, [[1, 2], [3, 4], [5, 6]])
    expected = Matrex.new(2, 2, [[22, 28], [49, 64]])

    assert Matrex.dot_tn(first, second) == expected
  end

  test "#dot_tn raises when sizes do not match" do
    first = Matrex.new(3, 2, [[1, 4], [2, 5], [3, 6]])
    second = Matrex.new(2, 4, [[2, 2, 3, 5], [3, 5, 6, 7]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
      Matrex.dot_tn(first, second)
    end
  end

  test "#first returns the first element of the matrix" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])

    assert Matrex.first(matrix) == 1
  end

  test "#size returns the size of the matrix" do
    matrix = Matrex.new(2, 3, [[4, 8, 22], [20, 0, 9]])

    assert Matrex.size(matrix) == {2, 3}
  end

  test "#inspect displays a matrix visualization to stdout" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = "Rows: 2 Columns: 3\n1 2 3\n4 5 6\n"

    output =
      capture_io(fn ->
        assert Matrex.inspect(matrix) == matrix
      end)

    assert output == expected
  end

  test "#max returns the maximum element from the matrix" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = 6

    assert Matrex.max(matrix) == expected
  end

  test "#multiply performs elementwise multiplication of two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new(2, 3, [[5, 4, 3], [12, 20, 36]])

    assert Matrex.multiply(first, second) == expected
  end

  test "#multiply raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 2, [[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
      Matrex.multiply(first, second)
    end
  end

  test "#multiply multiplies matrix element by a scalar" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    scalar = 2
    expected = Matrex.new(2, 3, [[2, 4, 6], [8, 10, 12]])

    assert Matrex.multiply(matrix, scalar) == expected
  end

  test "#neg negates matrix" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[-1, -2, -3], [-4, -5, -6]])

    assert Matrex.neg(matrix) == expected
  end

  test "#row returns row of the matrix" do
    matrix =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    assert Matrex.row(matrix, 4) == Matrex.new([[9, 11, 18, 25, 2]])
  end

  test "#row_as_list returns row of a matrix" do
    matrix =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    assert Matrex.row_to_list(matrix, 3) == [3, 10, 12, 19, 21]
  end

  test "#set changes value of one element" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[1, 2, 99], [4, 5, 6]])

    assert Matrex.set(matrix, 1, 3, 99) == expected
  end

  test "#substract substracts two matrices" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[-4, 0, 2], [1, 1, 0]])

    assert Matrex.substract(first, second) == expected
  end

  test "#substract raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/Matrices sizes mismatch./, fn ->
      Matrex.substract(first, second)
    end
  end

  test "#substract_inverse substracts the second matrix from the first" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new([[4, 0, -2], [-1, -1, 0]])

    assert Matrex.substract_inverse(first, second) == expected
  end

  test "#substract substracts matrix from scalar" do
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[0, -1, -2], [-3, -4, -5]])

    assert Matrex.substract(1, input) == expected
  end

  test "#sum/1 returns the sum of all elements in the matrix" do
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = 21

    assert Matrex.sum(input) == expected
  end

  test "#sum works precisely with big numbers" do
    input = Matrex.fill(1000, 10_000)
    expected = 1000 * 1000 * 10_000

    assert Matrex.sum(input) == expected
  end

  test "#to_list returs whole matrix as a list" do
    matrex = Matrex.new("1 2 3; 4 5 6; 7 8 9;")
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert Matrex.to_list(matrex) == expected
  end

  test "#to_list_of_lists returs whole matrix as a list" do
    matrex = Matrex.magic(3)
    expected = [[8.0, 1.0, 6.0], [3.0, 5.0, 7.0], [4.0, 9.0, 2.0]]
    assert Matrex.to_list_of_lists(matrex) == expected
  end

  test "#transpose transposes a matrix" do
    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new(3, 2, [[1, 4], [2, 5], [3, 6]])

    assert Matrex.transpose(input) == expected
  end

  test "#inspect/1 inspects matrix" do
    matrix = Matrex.eye(5)

    assert Matrex.Inspect.do_inspect(matrix) ==
             "\e[0m#Matrex[\e[33m5\e[0m×\e[33m5\e[0m]\n\e[0m┌                                         ┐\n│\e[33m     1.0\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[0m │\n│\e[33m\e[38;5;102m     0.0\e[33m     1.0\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[0m │\n│\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m     1.0\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[0m │\n│\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m     1.0\e[38;5;102m     0.0\e[33m\e[0m │\n│\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m\e[38;5;102m     0.0\e[33m     1.0 \e[0m│\n└                                         ┘"
  end

  test "#inspect/1 inspects matrix with infinities and NaNs" do
    one = Matrex.new("1 0 -1; 1 2 3; 4 5 6")
    two = Matrex.new("0 0 0; 0 1 0; 1 0 1")
    result = Matrex.divide(one, two)

    assert Matrex.Inspect.do_inspect(result) ==
             "\e[0m#Matrex[\e[33m3\e[0m×\e[33m3\e[0m]\n\e[0m┌                         ┐\n│\e[33m\e[36m     ∞  \e[33m\e[31m    NaN \e[33m\e[36m    -∞  \e[33m\e[0m │\n│\e[33m\e[36m     ∞  \e[33m     2.0\e[36m     ∞  \e[33m\e[0m │\n│\e[33m     4.0\e[36m     ∞  \e[33m     6.0 \e[0m│\n└                         ┘"
  end

  test "#inspect/1 inspects large matrix, skipping rows" do
    matrix = Matrex.magic(100)

    assert Matrex.Inspect.do_inspect(matrix) ==
             "\e[0m#Matrex[\e[33m100\e[0m×\e[33m100\e[0m]\n\e[0m┌                                                                                     ┐\n│\e[33m     1.0  9999.0  9998.0     4.0     5.0\e[0m  … \e[33m    96.0    97.0  9903.0  9902.0   100.0\e[0m │\n│\e[33m   9.9e3   102.0   103.0  9897.0  9896.0\e[0m  … \e[33m  9805.0  9804.0   198.0   199.0  9801.0\e[0m │\n│\e[33m   9.8e3   202.0   203.0  9797.0  9796.0\e[0m  … \e[33m  9705.0  9704.0   298.0   299.0  9701.0\e[0m │\n│\e[33m   301.0  9699.0  9698.0   304.0   305.0\e[0m  … \e[33m   396.0   397.0  9603.0  9602.0   400.0\e[0m │\n│\e[33m   401.0  9599.0  9598.0   404.0   405.0\e[0m  … \e[33m   496.0   497.0  9503.0  9502.0   500.0\e[0m │\n│\e[33m   9.5e3   502.0   503.0  9497.0  9496.0\e[0m  … \e[33m  9405.0  9404.0   598.0   599.0  9401.0\e[0m │\n│\e[33m   9.4e3   602.0   603.0  9397.0  9396.0\e[0m  … \e[33m  9305.0  9304.0   698.0   699.0  9301.0\e[0m │\n│\e[33m   701.0  9299.0  9298.0   704.0   705.0\e[0m  … \e[33m   796.0   797.0  9203.0  9202.0   800.0\e[0m │\n│\e[33m   801.0  9199.0  9198.0   804.0   805.0\e[0m  … \e[33m   896.0   897.0  9103.0  9102.0   900.0\e[0m │\n│\e[33m   9.1e3   902.0   903.0  9097.0  9096.0\e[0m  … \e[33m  9005.0  9004.0   998.0   999.0  9001.0\e[0m │\n│     ⋮       ⋮       ⋮       ⋮       ⋮    …      ⋮       ⋮       ⋮       ⋮       ⋮   │\n│\e[33m  9101.0   899.0   898.0  9104.0  9105.0\e[0m  … \e[33m  9196.0  9197.0   803.0   802.0   9.2e3\e[0m │\n│\e[33m  9201.0   799.0   798.0  9204.0  9205.0\e[0m  … \e[33m  9296.0  9297.0   703.0   702.0   9.3e3\e[0m │\n│\e[33m   700.0  9302.0  9303.0   697.0   696.0\e[0m  … \e[33m   605.0   604.0  9398.0  9399.0   601.0\e[0m │\n│\e[33m   600.0  9402.0  9403.0   597.0   596.0\e[0m  … \e[33m   505.0   504.0  9498.0  9499.0   501.0\e[0m │\n│\e[33m  9501.0   499.0   498.0  9504.0  9505.0\e[0m  … \e[33m  9596.0  9597.0   403.0   402.0   9.6e3\e[0m │\n│\e[33m  9601.0   399.0   398.0  9604.0  9605.0\e[0m  … \e[33m  9696.0  9697.0   303.0   302.0   9.7e3\e[0m │\n│\e[33m   300.0  9702.0  9703.0   297.0   296.0\e[0m  … \e[33m   205.0   204.0  9798.0  9799.0   201.0\e[0m │\n│\e[33m   200.0  9802.0  9803.0   197.0   196.0\e[0m  … \e[33m   105.0   104.0  9898.0  9899.0   101.0\e[0m │\n│\e[33m  9901.0    99.0    98.0  9904.0  9905.0\e[0m  … \e[33m  9996.0  9997.0     3.0     2.0   1.0e4 \e[0m│\n└                                                                                     ┘"
  end
end
