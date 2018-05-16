defmodule Matrex.Operators do
  @moduledoc """
  Overrides Kernel math operators for use with matrices.
  Use with caution.

  ## Usage

      iex> import IEx.Helpers, except: [t: 1] # Only in iex, conflicts with transpose function
      iex> import Matrex.Operators
      iex> import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]

      iex> m = random(5, 3)
      #Matrex[5×3]
      ┌                         ┐
      │ 0.51502 0.03132 0.94185 │
      │ 0.49434 0.93887 0.91102 │
      │ 0.70671 0.89428 0.28817 │
      │ 0.23771 0.37695 0.38214 │
      │ 0.37221 0.34008 0.19615 │
      └                         ┘
      iex> m * t(m) / eye(5) |> sigmoid()
      #Matrex[5×5]
      ┌                                         ┐
      │ 0.76012     1.0     1.0     1.0     1.0 │
      │     1.0 0.87608     1.0     1.0     1.0 │
      │     1.0     1.0 0.79935     1.0     1.0 │
      │     1.0     1.0     1.0 0.58531     1.0 │
      │     1.0     1.0     1.0     1.0 0.57265 │
      └                                         ┘

  """

  # Unary
  @doc false
  def -m, do: Matrex.neg(m)

  # Binary
  @doc false
  def a + b when is_number(a) and is_number(b), do: Kernel.+(a, b)
  def a + b, do: Matrex.add(a, b)

  @doc false
  def a - b when is_number(a) and is_number(b), do: Kernel.-(a, b)
  def a - b, do: Matrex.substract(a, b)

  @doc false
  def a * b when is_number(a) and is_number(b), do: Kernel.*(a, b)
  def a * b when is_number(a), do: Matrex.multiply(a, b)
  def a * b when is_number(b), do: Matrex.multiply(a, b)
  def a * b, do: Matrex.dot(a, b)

  @doc false
  def a / b when is_number(a) and is_number(b), do: Kernel./(a, b)
  def a / b, do: Matrex.divide(a, b)

  @doc false
  def a <|> b, do: Matrex.multiply(a, b)

  # Functions
  @doc "Transpose a matrix."
  def t(%Matrex{} = m), do: Matrex.transpose(m)
  @doc "Take logarithm of a matrix elementwise."
  def log(%Matrex{} = m), do: Matrex.apply(m, :log)

  @doc """
  Take sigmoid function of each element of matrix.
      sigmoid(x) = 1 / (1 + exp(-x))
  """
  def sigmoid(%Matrex{} = m), do: Matrex.apply(m, :sigmoid)

  @doc "See `Matrex.eye/1`"
  defdelegate eye(size), to: Matrex
  @doc "See `Matrex.ones/1`"
  defdelegate ones(size), to: Matrex
  @doc "See `Matrex.ones/2`"
  defdelegate ones(rows, cols), to: Matrex
  @doc "See `Matrex.random/1`"
  defdelegate random(size), to: Matrex
  @doc "See `Matrex.random/2`"
  defdelegate random(rows, cols), to: Matrex
  @doc "See `Matrex.set/4`"
  defdelegate set(matrex, row, col, val), to: Matrex
  @doc "See `Matrex.size/1`"
  defdelegate size(matrex), to: Matrex
  @doc "See `Matrex.scalar/1`"
  defdelegate scalar(matrex), to: Matrex
  @doc "See `Matrex.square/1`"
  defdelegate pow2(matrex), to: Matrex, as: :square
  @doc "See `Matrex.zeros/1`"
  defdelegate zeros(size), to: Matrex
  @doc "See `Matrex.zeros/2`"
  defdelegate zeros(rows, cols), to: Matrex
end
