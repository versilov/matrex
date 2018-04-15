defmodule MatrexTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO

  test "#add adds two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new(2, 3, [[6, 4, 4], [7, 9, 12]])

    assert Matrex.add(first, second) == expected
  end

  test "#add raises when sizes do not match" do
    first = Matrex.new([[1, 2, 3], [4, 5, 6]])
    second = Matrex.new([[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
      Matrex.add(first, second)
    end
  end

  test "#apply/2 applies a function/1 on each element of the matrix" do
    function = &(&1 + 1)
    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new(2, 3, [[2, 3, 4], [5, 6, 7]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/2 applies a function/2 on each element of the matrix" do
    function = fn element, index -> element + index end
    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new(2, 3, [[2, 4, 6], [8, 10, 12]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/2 applies a function/3 on each element of the matrix" do
    function = fn element, row_index, column_index ->
      element + row_index + column_index
    end

    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new(2, 3, [[3, 5, 7], [7, 9, 11]])

    assert Matrex.apply(input, function) == expected
  end

  test "#apply/3 applies a function on each element of the matrices" do
    function = &(&1 + &2)
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[2, 3, 4], [5, 6, 7]])
    expected = Matrex.new(2, 3, [[3, 5, 7], [9, 11, 13]])

    assert Matrex.apply(first, second, function) == expected
  end

  test "#argmax returns the index of the maximal element" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[8, 3, 4], [5, 6, 7]])
    third = Matrex.new(2, 3, [[8, 3, 4], [9, 6, 7]])

    assert Matrex.argmax(first) == 5
    assert Matrex.argmax(second) == 0
    assert Matrex.argmax(third) == 3
  end

  test "#at returns element at the given position" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])

    assert Matrex.at(matrix, 0, 2) == 3
  end

  test "#at raises when position is out of range" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, 0, 3)
    end
  end

  test "#divide divides two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 6], [9, 10, 18]])
    second = Matrex.new(2, 3, [[2, 2, 3], [3, 5, 6]])
    expected = Matrex.new(2, 3, [[0.5, 1, 2], [3, 2, 3]])

    assert Matrex.divide(first, second) == expected
  end

  test "#divide raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 2, [[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
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

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
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

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
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

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
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

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
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
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = "Rows: 2 Columns: 3\n1 2 3\n4 5 6\n"

    output =
      capture_io(fn ->
        assert Matrex.inspect(matrix) == matrix
      end)

    assert output == expected
  end

  test "#magic returns magic square of the given size" do
    magic5 =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    magic4 =
      Matrex.new([
        [1, 15, 14, 4],
        [12, 6, 7, 9],
        [8, 10, 11, 5],
        [13, 3, 2, 16]
      ])

    assert Matrex.magic(5) == magic5
    assert Matrex.max(magic5) == 25
    assert Matrex.sum(magic5) == 25 * (25 + 1) / 2

    assert Matrex.magic(4) == magic4
    assert Matrex.max(magic4) == 4 * 4
    assert Matrex.sum(magic4) == 4 * 4 * (4 * 4 + 1) / 2
  end

  test "#magic returns square with magic properties of arbitrary size" do
    n = 75

    magicN = Matrex.magic(n)
    assert Matrex.max(magicN) == n * n
    assert Matrex.sum(magicN) == n * n * div(n * n + 1, 2)
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

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
      Matrex.multiply(first, second)
    end
  end

  test "#multiply_with_scalar multiplies matrix element by a scalar" do
    matrix = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    scalar = 2
    expected = Matrex.new(2, 3, [[2, 4, 6], [8, 10, 12]])

    assert Matrex.multiply_with_scalar(matrix, scalar) == expected
  end

  test "#new creates a new matrix initialized by a function" do
    rows = 2
    columns = 3
    function = fn -> 1 end

    expected = <<
      2::float-little-32,
      3::float-little-32,
      1::float-little-32,
      1::float-little-32,
      1::float-little-32,
      1::float-little-32,
      1::float-little-32,
      1::float-little-32
    >>

    assert Matrex.new(rows, columns, function) == expected
  end

  test "#new creates a new matrix initialized by a list" do
    rows = 2
    columns = 3
    list = [[1, 2, 3], [4, 5, 6]]

    expected = <<
      2::float-little-32,
      3::float-little-32,
      1::float-little-32,
      2::float-little-32,
      3::float-little-32,
      4::float-little-32,
      5::float-little-32,
      6::float-little-32
    >>

    assert Matrex.new(rows, columns, list) == expected
  end

  test "#new creates a new matrix initialized by a list, without rows and cols specification" do
    list = [[1, 2, 3, 3], [4, 5, 6, 6]]

    expected = <<
      2::float-little-32,
      4::float-little-32,
      1::float-little-32,
      2::float-little-32,
      3::float-little-32,
      3::float-little-32,
      4::float-little-32,
      5::float-little-32,
      6::float-little-32,
      6::float-little-32
    >>

    assert Matrex.new(list) == expected
  end

  test "#substract substracts two matrices" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new(2, 3, [[-4, 0, 2], [1, 1, 0]])

    assert Matrex.substract(first, second) == expected
  end

  test "#substract raises when sizes do not match" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 2, [[5, 2], [3, 4]])

    assert_raise ErlangError, ~r/:matrices_size_mismatch/, fn ->
      Matrex.substract(first, second)
    end
  end

  test "#substract_inverse substracts the second matrix from the first" do
    first = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    second = Matrex.new(2, 3, [[5, 2, 1], [3, 4, 6]])
    expected = Matrex.new(2, 3, [[4, 0, -2], [-1, -1, 0]])

    assert Matrex.substract_inverse(first, second) == expected
  end

  test "#sum/1 returns the sum of all elements in the matrix" do
    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = 21

    assert Matrex.sum(input) == expected
  end

  test "#sum works precisely with big numbers" do
    input = Matrex.fill(1000, 10_000)
    expected = 1000 * 1000 * 10_000

    assert Matrex.sum(input) == expected
  end

  test "#transpose transposes a matrix" do
    input = Matrex.new(2, 3, [[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new(3, 2, [[1, 4], [2, 5], [3, 6]])

    assert Matrex.transpose(input) == expected
  end
end
