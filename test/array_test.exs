defmodule ArrayTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO
  import Matrex.Array
  alias Matrex.Array

  [:byte, :int16, :int32, :int64, :float32, :float64]
  |> Enum.each(fn type ->
    # quote do
    test "#add sums two arrays of #{type} of the same shape" do
      a = new([1, 2, 3, 4, 5, 6], {3, 2}, unquote(type))
      b = new([2, 3, 4, 5, 6, 7], {3, 2}, unquote(type))

      expected = new([3, 5, 7, 9, 11, 13], {3, 2}, unquote(type))

      c = add(a, b)

      assert c == expected
    end

    # end
  end)

  test "#add sums two arrays of bytes" do
    a = reshape(1..12, {3, 4}, :byte)
    b = reshape(12..1, {3, 4}, :byte)

    expected = fill(13, {3, 4}, :byte)

    assert add(a, b) == expected
  end

  test "#inspect shows array in console" do
    a = reshape(1..9, {3, 3}, :byte)

    expected = "1 2 3 4 5 6 7 8 9\n"

    output =
      capture_io(fn ->
        assert Array.inspect(a) == a
      end)

    assert output == expected
  end

  test "#new creates new array from list" do
    a = new([1, 2, 3, 4, 5, 6], {2, 3})
    assert at(a, 2, 3) == 6
  end

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
    assert strides({3, 4}, :float32) == {4 * 4, 4}
    assert strides({10, 5, 5}, :float32) == {25 * 4, 5 * 4, 4}
    assert strides({10, 5, 5}, :float32) == {25 * 4, 5 * 4, 4}
  end

  test "#transpose transposes 2-d array" do
    a = reshape(1..12, {3, 4}, :byte)
    t = transpose(a)
    assert at(t, 1, 2) == at(a, 2, 1)
    assert at(t, 1, 3) == at(a, 3, 1)
    assert at(t, 2, 2) == at(a, 2, 2)
  end

  test "#zeros creates 3-dim array of floats" do
    a = zeros({10, 28, 28}, :float32)
    assert at(a, {5, 10, 15}) == 0.0
  end

  test "#zeros creates 3-dim array of doubles" do
    a = zeros({10, 28, 28}, :float64)
    assert at(a, {5, 10, 15}) == 0.0
  end
end
