defmodule ArrayTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO
  import Matrex.Array
  alias Matrex.Array

  [:byte, :int16, :int32, :int64, :float32, :float64]
  |> Enum.each(fn type ->
    test "#add sums two arrays of #{type} of the same shape" do
      a = new([1, 2, 3, 4, 5, 6], {3, 2}, unquote(type))
      b = new([2, 3, 4, 5, 6, 7], {3, 2}, unquote(type))

      expected = new([3, 5, 7, 9, 11, 13], {3, 2}, unquote(type))

      c = add(a, b)

      assert c == expected
    end

    test "#add sums array of #{type} and a scalar" do
      a = new([1, 2, 3, 4, 5, 6], {3, 2}, unquote(type))
      scalar = 3.0

      expected = new([4, 5, 6, 7, 8, 9], {3, 2}, unquote(type))

      assert add(a, scalar) == expected
    end

    test "#multiply multiplies two arrays of #{type} of the same shape elementwise" do
      a = new([1, 2, 3, 4, 5, 6], {3, 2}, unquote(type))
      b = new([2, 3, 4, 5, 6, 7], {3, 2}, unquote(type))

      expected = new([2, 6, 12, 20, 30, 42], {3, 2}, unquote(type))

      c = multiply(a, b)

      assert c == expected
    end

    test "#sum sums elements of array of #{type}" do
      a = new([1, 2, 3, 4, 5, 6], {3, 2}, unquote(type))

      expected = 21

      assert sum(a) == expected
    end
  end)

  test "#apply applies function to bool array" do
    ab = new([1, 0, 1, 0, 0, 1], {3, 2}, :bool)
    expected = new([0, 1, 0, 1, 1, 0], {3, 2}, :bool)

    # Invert bool array with apply/2
    assert Array.apply(ab, fn
             1 -> 0
             0 -> 1
           end) == expected
  end

  test "#apply applies function with coords to bool array" do
    ab = new([1, 0, 1, 0, 0, 1, 0, 1], {2, 2, 2}, :bool)
    expected = new([0, 0, 0, 0, 0, 1, 0, 1], {2, 2, 2}, :bool)

    # Invert bool array's only first row with apply/2
    assert Array.apply(ab, fn
             _, {1, _c, _z} -> 0
             x, _ -> x
           end) == expected
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

  test "#new creates bool array from list" do
    a = new([true, true, false, true, true, false, false, true], {2, 4}, :bool)
    assert at(a, 2, 2) == 0
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

  test "#to_type/2 creates array converted to another type" do
    a = new([1.5, 1.3, 2.7, 3.33], {2, 2}, :float64)
    b = to_type(a, :int16)

    assert at(b, {1, 2}) == 1
    assert at(b, {2, 2}) == 3
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
