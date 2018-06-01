defmodule ArrayTest do
  use ExUnit.Case, async: true
  import Matrex.Array

  test "#random creates array of random values" do
    a = random({10, 25, 25})
    assert at(a, {5, 5, 5}) != at(a, {5, 5, 6})
  end

  test "#reshape changes shape of existing array" do
    a = reshape(1..12, {3, 4})
    assert shape(a) == {3, 4}
    r = reshape(a, {2, 6})
    assert shape(r) == {2, 6}
    assert at(a, 2, 3) == at(r, 2, 1)
  end

  test "#reshape creates array from range" do
    a = reshape(1..9, {3, 3})
    assert at(a, {3, 3}) == 9
  end

  test "#reshape creates byte array from range" do
    a = reshape(1..12, {4, 3}, :byte)
    assert at(a, 3, 2) == 8
  end

  test "#strides generates strides tuple" do
    assert strides({3, 4}, :float) == {4 * 4, 4}
    assert strides({10, 5, 5}, :float) == {25 * 4, 5 * 4, 4}
    assert strides({10, 5, 5}, :float) == {25 * 4, 5 * 4, 4}
  end

  test "#zeros creates 3-dim array of floats" do
    a = zeros({10, 28, 28}, :float)
    assert at(a, {5, 10, 15}) == 0.0
  end

  test "#zeros creates 3-dim array of doubles" do
    a = zeros({10, 28, 28}, :double)
    assert at(a, {5, 10, 15}) == 0.0
  end
end
