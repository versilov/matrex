defmodule MatrexBench.RandomMatrix do
  @doc """
  Generates a random matrix just so we can test large matrices
  """
  @spec random(integer, integer, integer) :: [[number]]
  def random(rows, cols, max) do
    Matrex.new(rows, cols, fn -> :rand.uniform(max) end)
  end
end

defmodule MatrexBench do
  @moduledoc """
  Benchfella module to compare matrix operations performance with a115/exmatrix

  Below are results for MacBookPro, 1 core used, 16 GB memory.
  ExLearn shows about 1 000 times better performance in dot product
  and is 20 times faster in transposing.

  ## ExMatrix
  ```


  benchmark name                iterations   average time
  transpose a 100x100 matrix          5000   763.27 µs/op
  transpose a 200x200 matrix          1000   2704.82 µs/op
  transpose a 400x400 matrix           100   13543.31 µs/op
  50x50 matrices dot product            20   79386.05 µs/op
  100x100 matrices dot product           2   615565.00 µs/op
  200x200 matrices dot product           1   4383168.00 µs/op
  400x400 matrices dot product           1   34386453.00 µs/op
  ```


  ## Matrex
  ```


  ## MatrexBench
  benchmark name                   iterations   average time
  transpose a 50x50 matrix             500000   3.32 µs/op
  50x50 matrices dot product           500000   6.61 µs/op
  transpose a 100x100 matrix           100000   12.47 µs/op
  300x300 zeros matrix creation        100000   12.99 µs/op
  300x300 eye matrix creation          100000   14.03 µs/op
  300x300 fill matrix                  100000   16.69 µs/op
  100x100 matrices dot product          50000   38.44 µs/op
  transpose a 200x200 matrix            50000   66.44 µs/op
  200x200 matrices dot product          10000   135.12 µs/op
  400x400 matrix sum                    10000   154.72 µs/op
  transpose a 400x400 matrix            10000   177.85 µs/op
  400x400 matrices dot product           5000   735.95 µs/op
  300x300 random matrix transpose        1000   1872.40 µs/op
  400x400 matrix to list                  500   4417.13 µs/op
  400x400 matrix to list of lists         500   5357.48 µs/op
  1000x1000 matrices dot product          200   8669.23 µs/op
  ```
  """

  use Benchfella
  import Matrex

  @random_a random(50)
  @random_b random(50)
  @random_a_large random(100)
  @random_b_large random(100)
  @random_a_qlarge random(200)
  @random_b_qlarge random(200)
  @random_a_vlarge random(400)
  @random_b_vlarge random(400)
  @random_a_xlarge random(1000)
  @random_b_xlarge random(1000)
  # @random_a random(50, 50, 100)
  # @random_b random(50, 50, 100)
  # @random_a_large random(100, 100, 100)
  # @random_b_large random(100, 100, 100)
  # @random_a_qlarge random(200, 200, 100)
  # @random_b_qlarge random(200, 200, 100)
  # @random_a_vlarge random(400, 400, 100)
  # @random_b_vlarge random(400, 400, 100)

  bench "transpose a 50x50 matrix" do
    transpose(@random_a)
  end

  bench "transpose a 100x100 matrix" do
    transpose(@random_a_large)
  end

  bench "transpose a 200x200 matrix" do
    transpose(@random_a_qlarge)
  end

  bench "transpose a 400x400 matrix" do
    transpose(@random_a_vlarge)
  end

  bench "50x50 matrices dot product" do
    dot(@random_a, @random_b)
  end

  bench "100x100 matrices dot product" do
    dot(@random_a_large, @random_b_large)
  end

  bench "200x200 matrices dot product" do
    dot(@random_a_qlarge, @random_b_qlarge)
  end

  bench "400x400 matrices dot product" do
    dot(@random_a_vlarge, @random_b_vlarge)
  end

  bench "1000x1000 matrices dot product" do
    dot(@random_a_xlarge, @random_b_xlarge)
  end

  bench "400x400 matrix exponent" do
    Matrex.apply(@random_a_vlarge, :exp)
  end

  bench "300x300 zeros matrix creation" do
    zeros(300)
  end

  bench "400x400 random matrix creation" do
    random(400)
    0
  end

  bench "300x300 fill matrix" do
    fill(300, 0)
  end

  bench "300x300 eye matrix creation" do
    eye(300)
  end

  bench "400x400 matrix sum" do
    sum(@random_a_vlarge)
  end

  bench "1000x1000 matrix add" do
    add(@random_a_xlarge, @random_b_xlarge)
  end

  bench "400x400 matrix to list" do
    to_list(@random_a_vlarge)
  end

  bench "400x400 matrix to list of lists" do
    to_list_of_lists(@random_a_vlarge)
  end

  bench "get element of a matrix" do
    at(@random_a_vlarge, :rand.uniform(399), :rand.uniform(399))
    0
  end
end
