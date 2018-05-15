defmodule Matrex.Operators do
  @moduledoc """
  Overrides Kernel math operators for use with matrices.
  Use with caution.

  ## Usage

      iex> import Matrex.Operators
      iex> import Kernel, except: [-:1, +: 2, -: 2, *: 2, /: 2, ".":2]
  """

  # Unary
  def -m, do: Matrex.neg(m)

  # Binary
  def a + b when is_number(a) and is_number(b), do: Kernel.+(a, b)
  def a + b, do: Matrex.add(a, b)
  def a - b when is_number(a) and is_number(b), do: Kernel.-(a, b)
  def a - b, do: Matrex.substract(a, b)
  def a * b when is_number(a) and is_number(b), do: Kernel.*(a, b)
  def a * b when is_number(a), do: Matrex.multiply(a, b)
  def a * b when is_number(b), do: Matrex.multiply(a, b)
  def a * b, do: Matrex.dot(a, b)
  def a / b when is_number(a) and is_number(b), do: Kernel./(a, b)
  def a / b, do: Matrex.divide(a, b)
  def a <|> b, do: Matrex.multiply(a, b)

  # Functions
  def t(%Matrex{} = m), do: Matrex.transpose(m)
  def log(%Matrex{} = m), do: Matrex.apply(m, :log)
  def sigmoid(%Matrex{} = m), do: Matrex.apply(m, :sigmoid)

  defdelegate eye(size), to: Matrex
  defdelegate ones(size), to: Matrex
  defdelegate ones(rows, cols), to: Matrex
  defdelegate random(size), to: Matrex
  defdelegate random(rows, cols), to: Matrex
  defdelegate set(matrex, row, col, val), to: Matrex
  defdelegate size(matrex), to: Matrex
  defdelegate scalar(matrex), to: Matrex
  defdelegate pow2(matrex), to: Matrex, as: :square
  defdelegate zeros(size), to: Matrex
  defdelegate zeros(rows, cols), to: Matrex
end
