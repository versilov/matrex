defmodule CreationTest do
  use ExUnit.Case, async: true

  import Matrex

  test "#eye creates a diagonal square matirx" do
    expected =
      Matrex.new([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
      ])

    assert Matrex.eye(5) == expected
  end

  test "#eye creates a diagonal square matrix of arbitrary value" do
    expected =
      Matrex.new([
        [7.39, 0, 0, 0, 0],
        [0, 7.39, 0, 0, 0],
        [0, 0, 7.39, 0, 0],
        [0, 0, 0, 7.39, 0],
        [0, 0, 0, 0, 7.39]
      ])

    assert Matrex.eye(5, 7.39) == expected
  end

  test "#fill fills new matrex with value" do
    expected = Matrex.new("7.53 7.53 7.53; 7.53 7.53 7.53; 7.53 7.53 7.53")
    assert Matrex.fill({3, 3}, 7.53) == expected
  end

  test "#fill fills matrix with special float value" do
    e = new("NegInf NegInf; NegInf NegInf")
    assert fill({2, 2}, :neg_inf) == e
  end

  test "#magic raises error, when too small is requested" do
    assert_raise ArgumentError, ~r/Magic square less than 3x3 is not possible./, fn ->
      Matrex.magic(2)
    end
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

    magic6 =
      Matrex.new([
        [5, 8, 36, 33, 13, 16],
        [6, 7, 34, 35, 14, 15],
        [28, 25, 17, 20, 12, 9],
        [26, 27, 18, 19, 10, 11],
        [24, 21, 4, 1, 32, 29],
        [22, 23, 2, 3, 30, 31]
      ])

    assert Matrex.magic(5) == magic5
    assert Matrex.max(magic5) == 25
    assert Matrex.sum(magic5) == 25 * (25 + 1) / 2

    assert Matrex.magic(4) == magic4
    assert Matrex.max(magic4) == 4 * 4
    assert Matrex.sum(magic4) == 4 * 4 * (4 * 4 + 1) / 2

    assert Matrex.magic(6) == magic6
    assert Matrex.max(magic6) == 6 * 6
    assert Matrex.sum(magic6) == 6 * 6 * (6 * 6 + 1) / 2
  end

  test "#magic returns square with magic properties of arbitrary size" do
    n = 75

    magicN = Matrex.magic(n)
    assert Matrex.max(magicN) == n * n
    assert Matrex.sum(magicN) == n * n * div(n * n + 1, 2)
  end

  test "#new creates matrix from text representation" do
    from_text =
      Matrex.new("""
        1.00000   0.10000   0.60000   1.10000
        1.00000   0.20000   0.70000   1.20000
        1.00000   0.30000   0.80000   1.30000
        1.00000   0.40000   0.90000   1.40000
        1.00000   0.50000   1.00000   1.50000
      """)

    expected =
      Matrex.new([
        [1.0, 0.1, 0.6, 1.1],
        [1.0, 0.2, 0.7, 1.2],
        [1.0, 0.3, 0.8, 1.3],
        [1.0, 0.4, 0.9, 1.4],
        [1.0, 0.5, 1.0, 1.5]
      ])

    assert from_text == expected
  end

  test "#new creates matirx from one-line text form" do
    from_line = Matrex.new("1;0;1;0;1")
    expected = Matrex.new([[1], [0], [1], [0], [1]])
    assert from_line == expected
  end

  test "#new creates a new matrix initialized by a function" do
    rows = 2
    columns = 3
    function = fn -> 1 end

    expected = %Matrex{
      data: <<
        1::float-little-32,
        1::float-little-32,
        1::float-little-32,
        1::float-little-32,
        1::float-little-32,
        1::float-little-32
      >>,
      shape: {rows, columns},
      strides: {12, 4},
      type: :float32
    }

    assert Matrex.new({rows, columns}, function) == expected
  end

  test "#new creates a new matrix initialized by a function with (row, col) arguments" do
    rows = 3
    columns = 3
    function = fn {row, col} -> row * col end

    expected = %Matrex{
      data: <<
        1::float-little-32,
        2::float-little-32,
        3::float-little-32,
        2::float-little-32,
        4::float-little-32,
        6::float-little-32,
        3::float-little-32,
        6::float-little-32,
        9::float-little-32
      >>,
      shape: {3, 3},
      strides: {12, 4},
      type: :float32
    }

    assert Matrex.new({rows, columns}, function) == expected
  end

  test "#new creates a new matrix initialized by a list of lists" do
    list = [[1, 2, 3], [4, 5, 6]]

    expected = %Matrex{
      data: <<
        1::float-little-32,
        2::float-little-32,
        3::float-little-32,
        4::float-little-32,
        5::float-little-32,
        6::float-little-32
      >>,
      shape: {2, 3},
      strides: {12, 4},
      type: :float32
    }

    assert Matrex.new(list) == expected
  end

  test "#new creates a new matrix from a list of lists of matrices" do
    m1 = Matrex.reshape(1..6, {2, 3})
    m2 = Matrex.reshape(7..12, {2, 3})
    m3 = Matrex.reshape(13..16, {2, 2})
    m4 = Matrex.reshape(17..20, {2, 2})

    expected =
      Matrex.new([[1, 2, 3, 13, 14], [4, 5, 6, 15, 16], [7, 8, 9, 17, 18], [10, 11, 12, 19, 20]])

    assert Matrex.new([[m1, m3], [m2, m4]]) == expected
  end

  test "#ones creates matrix filled with ones" do
    ones_matrix = Matrex.ones(7)

    assert ones_matrix ==
             Matrex.new([
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1]
             ])

    ones_not_square_matrix = Matrex.ones({3, 4})
    assert ones_not_square_matrix == Matrex.new([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
  end

  test "#random creates matrix of random values" do
    random_matrix = Matrex.random(10)
    assert Matrex.at(random_matrix, 5, 7) != Matrex.at(random_matrix, 5, 8)
  end

  test "#random does not generate the same matrix twice" do
    result = Matrex.divide(Matrex.random(100), Matrex.random(100))
    refute Matrex.contains?(result, 1.0)
  end

  test "#zeros/1 returns zero filled square matrix" do
    zero_matrix = Matrex.new([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert Matrex.zeros(3) == zero_matrix
  end

  test "#zeros/2 returns zero filled matrix" do
    zero_matrix = Matrex.new([[0, 0, 0], [0, 0, 0]])
    assert Matrex.zeros({2, 3}) == zero_matrix
  end
end
