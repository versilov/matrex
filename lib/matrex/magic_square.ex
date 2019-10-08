defmodule Matrex.MagicSquare do
  @moduledoc false

  # Magic square generation algorithms.

  @lux %{L: [4, 1, 2, 3], U: [1, 4, 2, 3], X: [1, 4, 3, 2]}

  def new(n) when n < 3, do: raise(ArgumentError, "Magic square less than 3x3 is not possible.")

  def new(n) when rem(n, 2) == 1 do
    for i <- 0..(n - 1) do
      for j <- 0..(n - 1),
          do: n * rem(i + j + 1 + div(n, 2), n) + rem(i + 2 * j + 2 * n - 5, n) + 1
    end
  end

  def new(n) when rem(n, 4) == 0 do
    n2 = n * n

    Enum.zip(1..n2, make_pattern(n))
    |> Enum.map(fn {i, p} -> if p, do: i, else: n2 - i + 1 end)
    |> Enum.chunk_every(n)
  end

  def new(n) when rem(n - 2, 4) == 0 do
    n2 = div(n, 2)
    oms = odd_magic_square(n2)
    mat = make_lux_matrix(n2)
    square = synthesis(n2, oms, mat)
    square
  end

  # zero beginning, it is 4 multiples.
  defp odd_magic_square(m) do
    for i <- 0..(m - 1),
        j <- 0..(m - 1),
        into: %{},
        do: {{i, j}, (m * rem(i + j + 1 + div(m, 2), m) + rem(i + 2 * j - 5 + 2 * m, m)) * 4}
  end

  defp make_lux_matrix(m) do
    center = div(m, 2)
    lux = List.duplicate(:L, center + 1) ++ [:U] ++ List.duplicate(:X, m - center - 2)

    for(
      {x, i} <- Enum.with_index(lux),
      j <- 0..(m - 1),
      into: %{},
      do: {{i, j}, x}
    )
    |> Map.put({center, center}, :U)
    |> Map.put({center + 1, center}, :L)
  end

  defp synthesis(m, oms, mat) do
    range = 0..(m - 1)

    Enum.reduce(range, [], fn i, acc ->
      {row0, row1} =
        Enum.reduce(range, {[], []}, fn j, {r0, r1} ->
          x = oms[{i, j}]
          [lux0, lux1, lux2, lux3] = @lux[mat[{i, j}]]
          {[x + lux0, x + lux1 | r0], [x + lux2, x + lux3 | r1]}
        end)

      [row0, row1 | acc]
    end)
  end

  defp make_pattern(n) do
    pattern =
      Enum.reduce(1..4, [true], fn _, acc ->
        acc ++ Enum.map(acc, &(!&1))
      end)
      |> Enum.chunk_every(4)

    for i <- 0..(n - 1),
        j <- 0..(n - 1),
        do: Enum.at(pattern, rem(i, 4)) |> Enum.at(rem(j, 4))
  end
end
