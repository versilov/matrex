defmodule Matrex.Operators do
  @moduledoc """
  Overrides Kernel math operators and adds common math functions shortcuts for use with matrices.
  Use with caution.

  ## Usage

      iex> import IEx.Helpers, except: [t: 1] # Only in iex, conflicts with transpose function
      iex> import Matrex.Operators
      iex> import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]
      iex> import Matrex

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
  def a - b, do: Matrex.subtract(a, b)

  @doc false
  def a * b when is_number(a) and is_number(b), do: Kernel.*(a, b)
  def a * b when is_number(a), do: Matrex.multiply(a, b)
  def a * b when is_number(b), do: Matrex.multiply(a, b)
  def a * b, do: Matrex.dot(a, b)

  @doc false
  def a / b when is_number(a) and is_number(b), do: Kernel./(a, b)
  def a / b, do: Matrex.divide(a, b)

  @doc "Element-wise matrices multiplication. The same as `Matrex.multiply/2`"
  def a <|> b, do: Matrex.multiply(a, b)

  # Define shortcuts for math funcions
  Enum.each(Matrex.math_functions_list(), fn f ->
    @doc "Applies C language #{f}(x) to each element of the matrix. See `Matrex.apply/2`"
    def unquote(f)(%Matrex{} = m), do: Matrex.apply(m, unquote(f))
  end)

  # Functions
  @doc "Transpose a matrix."
  defdelegate t(m), to: Matrex, as: :transpose
  @doc "See `Matrex.square/1`"
  defdelegate pow2(matrex), to: Matrex, as: :square
end
