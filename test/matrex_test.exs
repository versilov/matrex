defmodule MatrexTest do
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

    assert_raise FunctionClauseError, fn ->
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
      Matrex.at(matrix, {2, 5})
    end

    assert_raise ArgumentError, fn ->
      Matrex.at(matrix, {1, -1})
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

  test "#concat concatenates two matrices along columns" do
    first = Matrex.reshape(1..6, {2, 3})
    second = Matrex.reshape(1..4, {2, 2})
    expected = Matrex.new("1 2 3 1 2; 4 5 6 3 4")

    assert Matrex.concat(first, second) == expected
  end

  test "#concat concatenates list of matrices along columns" do
    z = Matrex.zeros(2)
    o = Matrex.ones(2)
    t = Matrex.fill(2, 2)

    expected = Matrex.new("0 0 1 1 2 2; 0 0 1 1 2 2")
    assert Matrex.concat([z, o, t]) == expected
  end

  test "#concat concatenates two matrices along rows" do
    first = Matrex.reshape(1..6, {3, 2})
    second = Matrex.reshape(1..4, {2, 2})
    expected = Matrex.new("1 2; 3 4; 5 6; 1 2; 3 4")

    assert Matrex.concat(first, second, :rows) == expected
  end

  test "#concat raises when sizes do not match" do
    first = Matrex.reshape(1..6, {3, 2})
    second = Matrex.reshape(1..6, {2, 3})

    assert_raise ArgumentError, fn ->
      Matrex.concat(first, second, :rows)
    end
  end

  test "#contains? checks if given element exists in the matrix" do
    matrix = Matrex.new("1 2 3; 4 5 6")
    assert Matrex.contains?(matrix, 3)
    assert Matrex.contains?(matrix, 6)
    refute Matrex.contains?(matrix, 0)
    refute Matrex.contains?(matrix, 10)
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

    assert_raise ErlangError, ~r/Matrices sizes mismatch./, fn ->
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

    assert_raise FunctionClauseError, fn ->
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

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
      Matrex.dot_and_add(first, second, third)
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

  test "#find returns position tuple of the element" do
    matrex = Matrex.reshape(1..100, {10, 10})
    assert Matrex.find(matrex, 11) == {2, 1}
    assert Matrex.find(matrex, 101) == nil
  end

  test "#find finds special float values" do
    matrex = Matrex.reshape([:neg_inf, 2, :nan, 4, :inf, 6], {2, 3})
    assert Matrex.find(matrex, :inf) == {2, 2}
    assert Matrex.find(matrex, :neg_inf) == {1, 1}
    assert Matrex.find(matrex, :nan) == {1, 3}
  end

  test "#first returns the first element of the matrix" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])

    assert Matrex.first(matrix) == 1
  end

  test "#list_of_rows returns row matrices list" do
    m = reshape(1..12, {6, 2})
    assert list_of_rows(m, 3..5) == [new([[5, 6]]), new([[7, 8]]), new([[9, 10]])]
  end

  test "#list_of_rows returns all rows when no range is given" do
    m = reshape(1..6, {3, 2})
    assert list_of_rows(m) == [new([[1, 2]]), new([[3, 4]]), new([[5, 6]])]
  end

  test "#max returns the maximum element from the matrix" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = 6

    assert Matrex.max(matrix) == expected
  end

  test "#max returns :inf" do
    matrix = Matrex.new([[1, :inf, 3], [4, :neg_inf, 6]])
    expected = :inf

    assert Matrex.max(matrix) == expected
  end

  test "#min returns the minimum element from the matrix" do
    matrix = Matrex.new([[1, 2, 0.5], [4, 5, 6]])
    expected = 0.5

    assert Matrex.min(matrix) == expected
  end

  test "#min returns :neg_inf" do
    matrix = Matrex.new([[1, 2, 0.5], [4, 5, :neg_inf]])
    expected = :neg_inf

    assert Matrex.min(matrix) == expected
  end

  test "#max_finite returns max finite element" do
    m = reshape([:inf, 2, 3, :inf], {2, 2})
    assert max_finite(m) == 3.0
  end

  test "#max_finite returns nil for totally infinite matrix" do
    m = fill(3, 3, :inf)
    assert max_finite(m) == nil
  end

  test "#min_finite returns min finite element" do
    m = reshape([:neg_inf, -2, :nan, 5], {2, 2})
    assert min_finite(m) == -2.0
  end

  test "#min_finite returns nil for totally infinite matrix" do
    m = fill(3, 3, :neg_inf)
    assert min_finite(m) == nil
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

    assert_raise ErlangError, ~r/Matrices sizes mismatch/, fn ->
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

  test "#normalize puts matrix values in [0, 1] range" do
    matrix = Matrex.reshape(1..12, {4, 3})

    expected =
      Matrex.new([
        [0.0, 0.09090909361839294, 0.1818181872367859],
        [0.27272728085517883, 0.3636363744735718, 0.4545454680919647],
        [0.5454545617103577, 0.6363636255264282, 0.7272727489471436],
        [0.8181818127632141, 0.9090909361839294, 1.0]
      ])

    assert Matrex.normalize(matrix) == expected
  end

  test "#resize scales down the dimensions of the matrix and interpolaties values" do
    m = Matrex.reshape(1..16, {4, 4})
    expected = Matrex.new([[1, 3], [9, 11]])

    assert Matrex.resize(m, 0.5) == expected
  end

  test "#resize scales up the dimensions of the matrix and interpolaties values" do
    m = Matrex.reshape(1..4, {2, 2})
    expected = Matrex.new([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

    assert Matrex.resize(m, 2) == expected
  end

  test "#reshape consumes any value, convertable to list" do
    assert Matrex.reshape(1..6, {2, 3}) == Matrex.new("1 2 3; 4 5 6")
    assert Matrex.reshape('abcd', {2, 2}) == Matrex.new("97 98; 99 100")
  end

  test "#reshape turns flat list into a matrix" do
    list = Enum.to_list(1..6)
    expected = Matrex.new("1 2; 3 4; 5 6")

    assert Matrex.reshape(list, {3, 2}) == expected
  end

  test "#reshape transforms list of matrices into a one big matrix" do
    list_of_matrices = [
      Matrex.fill(3, 1),
      Matrex.fill(3, 2),
      Matrex.fill(3, 3),
      Matrex.fill(3, 4)
    ]

    expected =
      Matrex.new("""
      1 1 1 2 2 2;
      1 1 1 2 2 2;
      1 1 1 2 2 2;
      3 3 3 4 4 4;
      3 3 3 4 4 4;
      3 3 3 4 4 4;
      """)

    assert Matrex.reshape(list_of_matrices, {2, 2}) == expected
  end

  test "#reshape respects special float values" do
    list = Enum.to_list(1..4) ++ [:nan, :neg_inf]
    expected = Matrex.new("1 2; 3 4; NaN NegInf")

    assert Matrex.reshape(list, {3, 2}) == expected
  end

  test "#reshape raises, when list length and shape do not match" do
    list = Enum.to_list(1..8)

    assert_raise ArgumentError, fn ->
      Matrex.reshape(list, {3, 2})
    end

    assert_raise ArgumentError, fn ->
      Matrex.reshape(list, {2, 5})
    end
  end

  test "#reshape converts Matrex to Matrex" do
    m = Matrex.reshape(1..12, {3, 4})
    expected = Matrex.new("1 2 3 4 5 6; 7 8 9 10 11 12")
    assert Matrex.reshape(m, {2, 6}) == expected
  end

  test "#reshape raises ArgumentError when matrices sizes do not match" do
    m = Matrex.reshape(1..12, {3, 4})

    assert_raise ArgumentError, fn ->
      Matrex.reshape(m, {3, 5})
    end
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

  test "#set can handle special values" do
    matrix = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[1, :nan, 3], [4, 5, 6]])

    assert Matrex.set(matrix, 1, 2, :nan) == expected
  end

  test "#size returns the size of the matrix" do
    matrix = Matrex.new([[4, 8, 22], [20, 0, 9]])

    assert Matrex.size(matrix) == {2, 3}
  end

  test "#submatrix returns part of original matrix" do
    matrix =
      Matrex.new([
        [16, 23, 5, 7, 14],
        [22, 4, 6, 13, 20],
        [3, 10, 12, 19, 21],
        [9, 11, 18, 25, 2],
        [15, 17, 24, 1, 8]
      ])

    expected =
      Matrex.new([
        [6, 13, 20],
        [12, 19, 21],
        [18, 25, 2]
      ])

    assert Matrex.submatrix(matrix, 2..4, 3..5) == expected
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

    assert_raise ErlangError, ~r/Matrices sizes mismatch./, fn ->
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

  test "#sum returns special float values" do
    input = Matrex.new([[:inf, 1.5, 0.0]])
    assert Matrex.sum(input) == :inf
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

  test "#to_row converts any matrix into a row matrix" do
    m = Matrex.magic(3)
    expected = Matrex.new([[8.0, 1.0, 6.0, 3.0, 5.0, 7.0, 4.0, 9.0, 2.0]])

    assert Matrex.to_row(m) == expected
  end

  test "#to_column converts any matrix into a column matrix" do
    m = Matrex.magic(3)
    expected = Matrex.new("8.0; 1.0; 6.0; 3.0; 5.0; 7.0; 4.0; 9.0; 2.0")

    assert Matrex.to_column(m) == expected
  end

  test "#transpose transposes a matrix" do
    input = Matrex.new([[1, 2, 3], [4, 5, 6]])
    expected = Matrex.new([[1, 4], [2, 5], [3, 6]])

    assert Matrex.transpose(input) == expected
  end

  test "#update updates element of a matrix with a function" do
    m = reshape(1..6, {3, 2})
    e = new("1 2; 3 16; 5 6")
    assert update(m, {2, 2}, fn x -> x * x end) == e
  end

  test "#update raises when position is out of bounds" do
    m = reshape(1..6, {3, 2})
    assert_raise ArgumentError, fn -> update(m, {2, 3}, fn x -> x * x end) end
  end
end
