defmodule GameOfLife do
  @moduledoc """
  Conway's game of life implementation with Matrex boolean array

  ## Example
      matrex$ iex -S mix
      iex> c "examples/game_of_life.exs"
      [GameOfLife]
      iex> GameOfLife.run(100, 100, 100)
  """
  alias Matrex.Array

  @doc """
  Run game of life.
    `width` — width of field
    `height` — height of field
    `iterations` — number of iterations to run game for.
  """
  def run(width, height, iterations) do
    field = Array.random({width, height}, :bool)

    IO.puts("#{hide_cursor()}")

    Enum.reduce(1..iterations, field, fn _i, fld ->
      IO.puts(IO.ANSI.home())
      Array.heatmap(fld)

      Array.apply(fld, fn x, {r, c} ->
        alive = Enum.sum(neighbours(fld, {r, c}))

        if alive == 3 or (alive == 2 and x == 1),
          do: 1,
          else: 0
      end)
    end)

    IO.puts("#{show_cursor()}")
  end

  defp neighbours(fld, {r, c}) do
    [
      cell_at(fld, r - 1, c - 1),
      cell_at(fld, r - 1, c),
      cell_at(fld, r - 1, c + 1),
      cell_at(fld, r, c - 1),
      cell_at(fld, r, c + 1),
      cell_at(fld, r + 1, c - 1),
      cell_at(fld, r + 1, c),
      cell_at(fld, r + 1, c + 1)
    ]
  end

  # Return zero, when we are out of range
  defp cell_at(%Array{shape: {rows, cols}}, r, c) when r < 1 or c < 1 or r > rows or c > cols,
    do: 0

  defp cell_at(fld, r, c), do: Array.at(fld, {r, c})

  defp hide_cursor(), do: "\e[?25l"
  defp show_cursor(), do: "\e[?25h"
end
