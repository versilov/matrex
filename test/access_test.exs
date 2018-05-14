defmodule AccessTest do
  use ExUnit.Case, async: true

  test "#[]/1 returns row of a matrix" do
    matrix = Matrex.new([[2, 23, 20], [2, 67, 9], [9, 18, 0]])
    assert matrix[2] == Matrex.new([[2, 67, 9]])
  end

  test "#[][] returns element of a matrix" do
    matrix = Matrex.new([[2, 23, 20], [2, 67, 9], [9, 18, 0]])
    assert matrix[3][2] == 18
  end

  test "#[a..b] returns subset of column matrix values" do
    matrix = Matrex.new("1;2;3;4;5;6;7;8")
    expected = Matrex.new("2;3;4")

    assert matrix[2..4] == expected
  end

  test "#[a..b] returns subset of row matrix values" do
    matrix = Matrex.new("1 2 3 4 5 6 7 8")
    expected = Matrex.new("2 3 4")

    assert matrix[2..4] == expected
  end

  test "#[a..b] returns subset of matrix rows" do
    matrix = Matrex.new("1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25")
    expected = Matrex.new("6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20")
    assert matrix[2..4] == expected
  end
end
