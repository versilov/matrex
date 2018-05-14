defmodule EnumerableTest do
  use ExUnit.Case, async: true

  setup_all do
    {:ok, matrex: Matrex.magic(7)}
  end

  test "#count counts all elements in the matrix", %{matrex: matrex} do
    assert Enum.count(matrex) == 49
  end

  test "#member? checks membership in a matrix", %{matrex: matrex} do
    assert Enum.member?(matrex, 30)
  end

  test "#sum sums all elements in a matrix", %{matrex: matrex} do
    assert Enum.sum(matrex) == 1_225
  end

  test "#sort returns sorted list of matrix elements", %{matrex: matrex} do
    assert Enum.sort(matrex) == Enum.to_list(1..49)
  end

  test "#slice gives part of matrix by range of indexes", %{matrex: matrex} do
    assert Enum.slice(matrex, 7..10) == [39.0, 48.0, 1.0, 10.0]
  end

  test "#slice gives part of matrix by offset and length", %{matrex: matrex} do
    assert Enum.slice(matrex, 7, 4) == [39.0, 48.0, 1.0, 10.0]
  end

  test "#max gives maximum element", %{matrex: matrex} do
    assert Enum.max(matrex) == 49
  end

  test "#any? checks elements of matrix", %{matrex: matrex} do
    assert Enum.any?(matrex, fn x -> :math.sqrt(x) == 7.0 end)
  end

  test "#reduce computes sum of matrix", %{matrex: matrex} do
    assert Enum.reduce(matrex, fn x, acc -> x + acc end) == 1_225
  end
end
