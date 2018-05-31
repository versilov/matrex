defmodule NumpyBench do
  @moduledoc """
  Benchmarks to compare Matrex agains NumPy.
  """
  use Benchfella
  import Matrex

  @a random(3_000)
  @b random(3_000)

  bench "dot(A, B)" do
    dot(@a, @b)
  end

  bench "divide(A, B)" do
    divide(@a, @b)
  end

  bench "add(A, B)" do
    add(@a, @b)
  end

  bench "sigmoid(A)" do
    Matrex.apply(@a, :sigmoid)
  end
end
