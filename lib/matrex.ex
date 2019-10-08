defmodule Matrex do
  @moduledoc """
  Performs fast operations on matrices using native C code and CBLAS library.

  ## Access behaviour

  Access behaviour is partly implemented for Matrex, so you can do:

  ```elixir

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> m[2][3]
      7.0
  ```
  Or even:
  ```elixir

      iex> m[1..2]
      #Matrex[2×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      └                         ┘
  ```

  There are also several shortcuts for getting dimensions of matrix:
  ```elixir

      iex> m[:rows]
      3

      iex> m[:size]
      {3, 3}
  ```
  calculating maximum value of the whole matrix:
  ```elixir

      iex> m[:max]
      9.0
  ```
  or just one of it's rows:
  ```elixir

      iex> m[2][:max]
      7.0
  ```
  calculating one-based index of the maximum element for the whole matrix:
  ```elixir

      iex> m[:argmax]
      8
  ```
  and a row:
  ```elixir

      iex> m[2][:argmax]
      3
  ```
  ## Inspect protocol

  Matrex implements `Inspect` and looks nice in your console:

  ![Inspect Matrex](https://raw.githubusercontent.com/versilov/matrex/master/docs/matrex_inspect.png)

  ## Math operators overloading

  `Matrex.Operators` module redefines `Kernel` math operators (+, -, *, / <|>) and
  defines some convenience functions, so you can write calculations code in more natural way.

  It should be used with great caution. We suggest using it only inside specific functions
  and only for increased readability, because using `Matrex` module functions, especially
  ones which do two or more operations at one call, are 2-3 times faster.

  ### Example

  ```elixir

      def lr_cost_fun_ops(%Matrex{} = theta, {%Matrex{} = x, %Matrex{} = y, lambda} = _params)
          when is_number(lambda) do
        # Turn off original operators
        import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]
        import Matrex.Operators
        import Matrex

        m = y[:rows]

        h = sigmoid(x * theta)
        l = ones(size(theta)) |> set(1, 1, 0.0)

        j = (-t(y) * log(h) - t(1 - y) * log(1 - h) + lambda / 2 * t(l) * pow2(theta)) / m

        grad = (t(x) * (h - y) + (theta <|> l) * lambda) / m

        {scalar(j), grad}
      end
  ```


  The same function, coded with module methods calls (2.5 times faster):

  ```elixir
      def lr_cost_fun(%Matrex{} = theta, {%Matrex{} = x, %Matrex{} = y, lambda} = _params)
          when is_number(lambda) do
        m = y[:rows]

        h = Matrex.dot_and_apply(x, theta, :sigmoid)
        l = Matrex.ones(theta[:rows], theta[:cols]) |> Matrex.set(1, 1, 0)

        regularization =
          Matrex.dot_tn(l, Matrex.square(theta))
          |> Matrex.scalar()
          |> Kernel.*(lambda / (2 * m))

        j =
          y
          |> Matrex.dot_tn(Matrex.apply(h, :log), -1)
          |> Matrex.subtract(
            Matrex.dot_tn(
              Matrex.subtract(1, y),
              Matrex.apply(Matrex.subtract(1, h), :log)
            )
          )
          |> Matrex.scalar()
          |> (fn
                :nan -> :nan
                x -> x / m + regularization
              end).()

        grad =
          x
          |> Matrex.dot_tn(Matrex.subtract(h, y))
          |> Matrex.add(Matrex.multiply(theta, l), 1.0, lambda)
          |> Matrex.divide(m)

        {j, grad}
      end
  ```

  ## Enumerable protocol

  Matrex implements `Enumerable`, so, all kinds of `Enum` functions are applicable:

  ```elixir

      iex> Enum.member?(m, 2.0)
      true

      iex> Enum.count(m)
      9

      iex> Enum.sum(m)
      45
  ```

  For functions, that exist both in `Enum` and in `Matrex` it's preferred to use Matrex
  version, beacuse it's usually much, much faster. I.e., for 1 000 x 1 000 matrix `Matrex.sum/1`
  and `Matrex.to_list/1` are 438 and 41 times faster, respectively, than their `Enum` counterparts.

  ## Saving and loading matrix

  You can save/load matrix with native binary file format (extra fast)
  and CSV (slow, especially on large matrices).

  Matrex CSV format is compatible with GNU Octave CSV output,
  so you can use it to exchange data between two systems.

  ### Example

  ```elixir

      iex> Matrex.random(5) |> Matrex.save("rand.mtx")
      :ok
      iex> Matrex.load("rand.mtx")
      #Matrex[5×5]
      ┌                                         ┐
      │ 0.05624 0.78819 0.29995 0.25654 0.94082 │
      │ 0.50225 0.22923 0.31941  0.3329 0.78058 │
      │ 0.81769 0.66448 0.97414 0.08146 0.21654 │
      │ 0.33411 0.59648 0.24786 0.27596 0.09082 │
      │ 0.18673 0.18699 0.79753 0.08101 0.47516 │
      └                                         ┘
      iex> Matrex.magic(5) |> Matrex.divide(Matrex.eye(5)) |> Matrex.save("nan.csv")
      :ok
      iex> Matrex.load("nan.csv")
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0     ∞       ∞       ∞       ∞   │
      │     ∞       4.0     ∞       ∞       ∞   │
      │     ∞       ∞      12.0     ∞       ∞   │
      │     ∞       ∞       ∞      25.0     ∞   │
      │     ∞       ∞       ∞       ∞       8.0 │
      └                                         ┘
  ```

  ## NaN and Infinity

  Float special values, like `:nan` and `:inf` live well inside matrices,
  can be loaded from and saved to files.
  But when getting them into Elixir they are transferred to `:nan`,`:inf` and `:neg_inf` atoms,
  because BEAM does not accept special values as valid floats.

  ```elixir
      iex> m = Matrex.eye(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     0.0     0.0 │
      │     0.0     1.0     0.0 │
      │     0.0     0.0     1.0 │
      └                         ┘

      iex> n = Matrex.divide(m, Matrex.zeros(3))
      #Matrex[3×3]
      ┌                         ┐
      │     ∞      NaN     NaN  │
      │    NaN      ∞      NaN  │
      │    NaN     NaN      ∞   │
      └                         ┘

      iex> n[1][1]
      :inf

      iex> n[1][2]
      :nan
  ```

  """

  alias Matrex.NIFs
  import Matrex.Guards

  @enforce_keys [:data]
  defstruct [:data]
  @type element :: number | :nan | :inf | :neg_inf
  @type index :: pos_integer
  @type matrex :: %Matrex{data: binary}
  @type t :: matrex

  # Size of matrix element (float) in bytes
  @element_size 4

  # Float special values in binary form
  @not_a_number <<0, 0, 192, 255>>
  @positive_infinity <<0, 0, 128, 127>>
  @negative_infinity <<0, 0, 128, 255>>

  @compile {:inline,
            add: 2,
            argmax: 1,
            at: 3,
            binary_to_float: 1,
            column_to_list: 2,
            contains?: 2,
            divide: 2,
            dot: 2,
            dot_and_add: 3,
            dot_nt: 2,
            dot_tn: 2,
            forward_substitute: 2,
            cholesky: 1,
            eye: 1,
            diagonal: 1,
            element_to_string: 1,
            fill: 3,
            fill: 2,
            first: 1,
            fetch: 2,
            float_to_binary: 1,
            max: 1,
            multiply: 2,
            ones: 2,
            ones: 1,
            parse_float: 1,
            random: 2,
            random: 1,
            reshape: 3,
            row_to_list: 2,
            row: 2,
            set: 4,
            size: 1,
            square: 1,
            subtract: 2,
            subtract_inverse: 2,
            sum: 1,
            to_list: 1,
            to_list_of_lists: 1,
            to_row: 1,
            to_column: 1,
            transpose: 1,
            update: 4,
            zeros: 2,
            zeros: 1}

  @behaviour Access

  defmacrop matrex_data(rows, columns, body) do
    quote do
      %Matrex{
        data: <<
          unquote(rows)::unsigned-integer-little-32,
          unquote(columns)::unsigned-integer-little-32,
          unquote(body)::binary
        >>
      }
    end
  end

  defmacrop matrex_data(rows, columns, body, data) do
    quote do
      %Matrex{
        data:
          <<
            unquote(rows)::unsigned-integer-little-32,
            unquote(columns)::unsigned-integer-little-32,
            unquote(body)::binary
          >> = unquote(data)
      }
    end
  end

  @impl Access
  def fetch(matrex, key)

  # Horizontal vector
  def fetch(matrex_data(1, _, _) = matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, at(matrex, 1, key)}

  # Vertical vector
  def fetch(matrex_data(_, 1, _) = matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, at(matrex, key, 1)}

  # Return a row
  def fetch(matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, row(matrex, key)}

  # Slice on horizontal vector
  def fetch(matrex_data(1, columns, data), a..b)
      when b > a and a > 0 and b <= columns do
    data = binary_part(data, (a - 1) * @element_size, (b - a + 1) * @element_size)
    {:ok, matrex_data(1, b - a + 1, data)}
  end

  def fetch(matrex_data(rows, columns, data), a..b)
      when b > a and a > 0 and b <= rows do
    data =
      binary_part(data, (a - 1) * columns * @element_size, (b - a + 1) * columns * @element_size)

    {:ok, matrex_data(b - a + 1, columns, data)}
  end

  def fetch(matrex_data(rows, _, _), :rows), do: {:ok, rows}
  def fetch(matrex_data(_, cols, _), :cols), do: {:ok, cols}
  def fetch(matrex_data(_, cols, _), :columns), do: {:ok, cols}
  def fetch(matrex_data(rows, cols, _), :size), do: {:ok, {rows, cols}}
  def fetch(matrex, :sum), do: {:ok, sum(matrex)}
  def fetch(matrex, :max), do: {:ok, max(matrex)}
  def fetch(matrex, :min), do: {:ok, min(matrex)}
  def fetch(matrex, :argmax), do: {:ok, argmax(matrex)}

  def get(%Matrex{} = matrex, key, default) do
    case fetch(matrex, key) do
      {:ok, value} -> value
      :error -> default
    end
  end

  @impl Access
  def pop(matrex_data(rows, columns, body), row)
      when is_integer(row) and row >= 1 and row <= rows do
    get =
      matrex_data(
        1,
        columns,
        binary_part(body, (row - 1) * columns * @element_size, columns * @element_size)
      )

    update =
      matrex_data(
        rows - 1,
        columns,
        binary_part(body, 0, (row - 1) * columns * @element_size) <>
          binary_part(body, row * columns * @element_size, (rows - row) * columns * @element_size)
      )

    {get, update}
  end

  def pop(%Matrex{} = matrex, _), do: {nil, matrex}

  # To silence warnings
  @impl Access
  def get_and_update(%Matrex{}, _row, _fun), do: raise("not implemented")

  defimpl Inspect do
    @doc false
    def inspect(%Matrex{} = matrex, opts) do
      columns =
        case opts.width do
          :infinity -> 80
          width -> width
        end

      Matrex.Inspect.do_inspect(matrex, columns, 21)
    end
  end

  defimpl Enumerable do
    # Matrix element size in bytes
    @element_size 4

    defmacrop matrex_data(rows, columns, data) do
      quote do
        %Matrex{
          data: <<
            unquote(rows)::unsigned-integer-little-32,
            unquote(columns)::unsigned-integer-little-32,
            unquote(data)::binary
          >>
        }
      end
    end

    @doc false
    def count(matrex_data(rows, cols, _data)), do: {:ok, rows * cols}

    @doc false
    def member?(%Matrex{} = matrex, element), do: {:ok, Matrex.contains?(matrex, element)}

    @doc false
    def slice(matrex_data(rows, cols, body)) do
      {:ok, rows * cols,
       fn start, length ->
         Matrex.binary_to_list(binary_part(body, start * @element_size, length * @element_size))
       end}
    end

    @doc false
    def reduce(matrex_data(_rows, _cols, body), acc, fun) do
      reduce_each(body, acc, fun)
    end

    defp reduce_each(_, {:halt, acc}, _fun), do: {:halted, acc}

    defp reduce_each(matrix, {:suspend, acc}, fun),
      do: {:suspended, acc, &reduce_each(matrix, &1, fun)}

    defp reduce_each(<<elem::binary-@element_size, rest::binary>>, {:cont, acc}, fun),
      do: reduce_each(rest, fun.(Matrex.binary_to_float(elem), acc), fun)

    defp reduce_each(<<>>, {:cont, acc}, _fun), do: {:done, acc}
  end

  @doc """
  Adds scalar to matrix.

  See `Matrex.add/4` for details.
  """
  @spec add(matrex, number) :: matrex
  @spec add(number, matrex) :: matrex
  def add(%Matrex{data: matrix} = _a, b) when is_number(b),
    do: %Matrex{data: NIFs.add_scalar(matrix, b)}

  def add(a, %Matrex{data: matrix} = _b) when is_number(a),
    do: %Matrex{data: NIFs.add_scalar(matrix, a)}

  @doc """
  Adds two matrices or scalar to each element of matrix. NIF.

  Can optionally scale any of the two matrices.

  C = αA + βB

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.add(Matrex.new([[1,2,3],[4,5,6]]), Matrex.new([[7,8,9],[10,11,12]]))
      #Matrex[2×3]
      ┌                         ┐
      │     8.0    10.0    12.0 │
      │    14.0    16.0    18.0 │
      └                         ┘

  Adding with scalar:

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.add(m, 1)
      #Matrex[3×3]
      ┌                         ┐
      │     9.0     2.0     7.0 │
      │     4.0     6.0     8.0 │
      │     5.0    10.0     3.0 │
      └                         ┘

  With scaling each matrix:

      iex> Matrex.add(Matrex.new("1 2 3; 4 5 6"), Matrex.new("3 2 1; 6 5 4"), 2.0, 3.0)
      #Matrex[2×3]
      ┌                         ┐
      │     11.0    10.0    9.0 │
      │     26.0    25.0   24.0 │
      └                         ┘
  """

  @spec add(matrex, matrex, number, number) :: matrex
  def add(
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32,
              _data1::binary>> = first
        },
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32,
              _data2::binary>> = second
        },
        alpha \\ 1.0,
        beta \\ 1.0
      )
      when is_number(alpha) and is_number(beta),
      do: %Matrex{data: NIFs.add(first, second, alpha, beta)}

  @doc """

  Applies given function to each element of the matrix and returns the matrex of results. NIF.

  If second argument is an atom, then applies C language math function.

  ## Example

      iex> Matrex.magic(5) |> Matrex.apply(:sigmoid)
      #Matrex[5×5]
      ┌                                         ┐
      │-0.95766-0.53283 0.28366  0.7539 0.13674 │
      │-0.99996-0.65364 0.96017 0.90745 0.40808 │
      │-0.98999-0.83907 0.84385  0.9887-0.54773 │
      │-0.91113 0.00443 0.66032  0.9912-0.41615 │
      │-0.75969-0.27516 0.42418  0.5403 -0.1455 │
      └                                         ┘

  The following math functions from C <math.h> are supported, and also a sigmoid function:

  ```elixir
    :exp, :exp2, :sigmoid, :expm1, :log, :log2, :sqrt, :cbrt, :ceil, :floor, :truncate, :round,
    :abs, :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
    :erf, :erfc, :tgamma, :lgamm
  ```


  If second argument is a function that takes one argument,
  then this function receives the element of the matrix.

  ## Example

      iex> Matrex.magic(5) |> Matrex.apply(&:math.cos/1)
      #Matrex[5×5]
      ┌                                         ┐
      │-0.95766-0.53283 0.28366  0.7539 0.13674 │
      │-0.99996-0.65364 0.96017 0.90745 0.40808 │
      │-0.98999-0.83907 0.84385  0.9887-0.54773 │
      │-0.91113 0.00443 0.66032  0.9912-0.41615 │
      │-0.75969-0.27516 0.42418  0.5403 -0.1455 │
      └                                         ┘


  If second argument is a function that takes two arguments,
  then this function receives the element of the matrix and its one-based index.


  ## Example

      iex> Matrex.ones(5) |> Matrex.apply(fn val, index -> val + index end)
      #Matrex[5×5]
      ┌                                         ┐
      │     2.0     3.0     4.0     5.0     6.0 │
      │     7.0     8.0     9.0    10.0    11.0 │
      │    12.0    13.0    14.0    15.0    16.0 │
      │    17.0    18.0    19.0    20.0    21.0 │
      │    22.0    23.0    24.0    25.0    26.0 │
      └                                         ┘

  If second argument is a function that takes three arguments,
  then this function receives the element of the matrix one-based row index and one-based
  column index of the element.

  ## Example

      iex> Matrex.ones(5) |> Matrex.apply(fn val, row, col -> val + row + col end)
      #Matrex[5×5]
      ┌                                         ┐
      │     3.0     4.0     5.0     6.0     7.0 │
      │     4.0     5.0     6.0     7.0     8.0 │
      │     5.0     6.0     7.0     8.0     9.0 │
      │     6.0     7.0     8.0     9.0    10.0 │
      │     7.0     8.0     9.0    10.0    11.0 │
      └                                         ┘

  """
  @math_functions [
    :exp,
    :exp2,
    :sigmoid,
    :expm1,
    :log,
    :log2,
    :sqrt,
    :cbrt,
    :ceil,
    :floor,
    :truncate,
    :round,
    :abs,
    :sin,
    :cos,
    :tan,
    :asin,
    :acos,
    :atan,
    :sinh,
    :cosh,
    :tanh,
    :asinh,
    :acosh,
    :atanh,
    :erf,
    :erfc,
    :tgamma,
    :lgamma
  ]

  @spec apply(
          matrex,
          atom
          | (element -> element)
          | (element, index -> element)
          | (element, index, index -> element)
        ) :: matrex
  def apply(%Matrex{data: data} = _matrix, function_atom)
      when function_atom in @math_functions do
    # {rows, cols} = size(matrix)

    %Matrex{
      data:
        if(
          true,
          # rows * cols < 100_000,
          do: NIFs.apply_math(data, function_atom),
          else: NIFs.apply_parallel_math(data, function_atom)
        )
    }
  end

  def apply(
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32,
              data::binary>>
        },
        function
      )
      when is_function(function, 1) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    %Matrex{data: apply_on_matrix(data, function, initial)}
  end

  def apply(matrex_data(rows, columns, data), function)
      when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>
    size = rows * columns
    %Matrex{data: apply_on_matrix(data, function, 1, size, initial)}
  end

  def apply(matrex_data(rows, columns, data), function)
      when is_function(function, 3) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    %Matrex{data: apply_on_matrix(data, function, 1, 1, columns, initial)}
  end

  defp apply_on_matrix(<<>>, _, accumulator), do: accumulator

  defp apply_on_matrix(<<value::float-little-32, rest::binary>>, function, accumulator) do
    new_value = function.(value)

    apply_on_matrix(rest, function, <<accumulator::binary, new_value::float-little-32>>)
  end

  defp apply_on_matrix(<<>>, _, _, _, accumulator), do: accumulator

  defp apply_on_matrix(
         <<value::float-little-32, rest::binary>>,
         function,
         index,
         size,
         accumulator
       ) do
    new_value = function.(value, index)

    apply_on_matrix(
      rest,
      function,
      index + 1,
      size,
      <<accumulator::binary, new_value::float-little-32>>
    )
  end

  defp apply_on_matrix(<<>>, _, _, _, _, accumulator), do: accumulator

  defp apply_on_matrix(
         <<value::float-little-32, rest::binary>>,
         function,
         row_index,
         column_index,
         columns,
         accumulator
       ) do
    new_value = function.(value, row_index, column_index)
    new_accumulator = <<accumulator::binary, new_value::float-little-32>>

    case column_index < columns do
      true ->
        apply_on_matrix(rest, function, row_index, column_index + 1, columns, new_accumulator)

      false ->
        apply_on_matrix(rest, function, row_index + 1, 1, columns, new_accumulator)
    end
  end

  @doc """
  Applies function to elements of two matrices and returns matrix of function results.

  Matrices must be of the same size.

  ## Example

      iex(11)> Matrex.apply(Matrex.random(5), Matrex.random(5), fn x1, x2 -> min(x1, x2) end)
      #Matrex[5×5]
      ┌                                         ┐
      │ 0.02025 0.15055 0.69177 0.08159 0.07237 │
      │ 0.03252 0.14805 0.03627  0.1733 0.58721 │
      │ 0.10865 0.49192 0.12166  0.0573 0.66522 │
      │ 0.13642 0.23838 0.14403 0.57151 0.12359 │
      │ 0.12877 0.12745 0.10933 0.27281 0.35957 │
      └                                         ┘
  """
  @spec apply(matrex, matrex, (element, element -> element)) :: matrex
  def apply(matrex_data(rows, columns, data1), matrex_data(rows, columns, data2), function)
      when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>
    %Matrex{data: apply_on_matrices(data1, data2, function, initial)}
  end

  defp apply_on_matrices(<<>>, <<>>, _, accumulator), do: accumulator

  defp apply_on_matrices(
         <<first_value::float-little-32, first_rest::binary>>,
         <<second_value::float-little-32, second_rest::binary>>,
         function,
         accumulator
       )
       when is_function(function, 2) do
    new_value = function.(first_value, second_value)
    new_accumulator = <<accumulator::binary, new_value::float-little-32>>

    apply_on_matrices(first_rest, second_rest, function, new_accumulator)
  end

  @doc """
  Returns one-based index of the biggest element. NIF.

  There is also `matrex[:argmax]` shortcut for this function.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.argmax(m)
      7

  """
  @spec argmax(matrex) :: index
  def argmax(%Matrex{data: data}), do: NIFs.argmax(data) + 1

  @doc """
  Get element of a matrix at given one-based (row, column) position.

  Negative or out of bound indices will raise an exception.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.at(m, 3, 2)
      9.0

  You can use `Access` behaviour square brackets for the same purpose,
  but it will be slower:

      iex> m[3][2]
      9.0

  """
  @spec at(matrex, index, index) :: element
  def at(matrex_data(rows, columns, data), row, col)
      when is_integer(row) and is_integer(col) do
    if row < 1 or row > rows, do: raise(ArgumentError, "row position out of range: #{row}")

    if col < 1 or col > columns, do: raise(ArgumentError, "column position out of range: #{col}")

    data
    |> binary_part(((row - 1) * columns + (col - 1)) * @element_size, @element_size)
    |> binary_to_float()
  end

  @doc false
  @spec binary_to_float(<<_::32>>) :: element | :nan | :inf | :neg_inf
  def binary_to_float(@not_a_number), do: :nan
  def binary_to_float(@positive_infinity), do: :inf
  def binary_to_float(@negative_infinity), do: :neg_inf
  def binary_to_float(<<val::float-little-32>>), do: val

  @doc false
  @spec binary_to_list(<<_::_*32>>) :: [element | NaN | Inf | NegInf]
  def binary_to_list(<<elem::binary-@element_size, rest::binary>>),
    do: [binary_to_float(elem) | binary_to_list(rest)]

  def binary_to_list(<<>>), do: []

  @doc """
  Get column of matrix as matrix (vector) in matrex form. One-based.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.column(m, 2)
      #Matrex[3×1]
      ┌         ┐
      │     1.0 │
      │     5.0 │
      │     9.0 │
      └         ┘

  """
  @spec column(matrex, index) :: matrex
  def column(matrex_data(rows, columns, data), col)
      when is_integer(col) and col > 0 and col <= columns do
    column = <<rows::unsigned-integer-little-32, 1::unsigned-integer-little-32>>

    data =
      Enum.map(0..(rows - 1), fn row ->
        binary_part(data, (row * columns + (col - 1)) * @element_size, @element_size)
      end)

    %Matrex{data: IO.iodata_to_binary([column | data])}
  end

  @doc """
  Get column of matrix as list of floats. One-based, NIF.


  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.column_to_list(m, 3)
      [6.0, 7.0, 2.0]

  """
  @spec column_to_list(matrex, index) :: [element]
  def column_to_list(%Matrex{data: matrix}, column) when is_integer(column) and column > 0,
    do: NIFs.column_to_list(matrix, column - 1)

  @doc """
  Concatenate list of matrices along columns.

  The number of rows must be equal.

  ## Example

      iex> Matrex.concat([Matrex.fill(2, 0), Matrex.fill(2, 1), Matrex.fill(2, 2)])                #Matrex[2×6]
      ┌                                                 ┐
      │     0.0     0.0     1.0     1.0     2.0     2.0 │
      │     0.0     0.0     1.0     1.0     2.0     2.0 │
      └                                                 ┘

  """
  @spec concat([matrex]) :: matrex
  def concat([%Matrex{} | _] = list_of_ma), do: Enum.reduce(list_of_ma, &Matrex.concat(&2, &1))

  @doc """
  Concatenate two matrices along rows or columns. NIF.

  The number of rows or columns must be equal.

  ## Examples

      iex> m1 = Matrex.new([[1, 2, 3], [4, 5, 6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> m2 = Matrex.new([[7, 8, 9], [10, 11, 12]])
      #Matrex[2×3]
      ┌                         ┐
      │     7.0     8.0     9.0 │
      │    10.0    11.0    12.0 │
      └                         ┘
      iex> Matrex.concat(m1, m2)
      #Matrex[2×6]
      ┌                                                 ┐
      │     1.0     2.0     3.0     7.0     8.0     9.0 │
      │     4.0     5.0     6.0    10.0    11.0    12.0 │
      └                                                 ┘
      iex> Matrex.concat(m1, m2, :rows)
      #Matrex[4×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      │    10.0    11.0    12.0 │
      └                         ┘
  """
  @spec concat(matrex, matrex, :columns | :rows) :: matrex
  def concat(matrex1, matrex2, type \\ :columns)

  def concat(
        %Matrex{
          data:
            <<
              rows1::unsigned-integer-little-32,
              _rest1::binary
            >> = first
        },
        %Matrex{
          data:
            <<
              rows2::unsigned-integer-little-32,
              _rest2::binary
            >> = second
        },
        :columns
      )
      when rows1 == rows2,
      do: %Matrex{data: Matrex.NIFs.concat_columns(first, second)}

  def concat(matrex_data(rows1, columns, data1), matrex_data(rows2, columns, data2), :rows) do
    matrex_data(rows1 + rows2, columns, data1 <> data2)
  end

  def concat(matrex_data(rows1, columns1, _data1), matrex_data(rows2, columns2, _data2), type) do
    raise(
      ArgumentError,
      "Cannot concat: #{rows1}×#{columns1} does not fit with #{rows2}×#{columns2} along #{type}."
    )
  end

  @doc """
  Checks if given element exists in the matrix.

  ## Example

      iex> m = Matrex.new("1 NaN 3; Inf 10 23")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0    NaN      3.0 │
      │     ∞      10.0    23.0 │
      └                         ┘
      iex> Matrex.contains?(m, 1.0)
      true
      iex> Matrex.contains?(m, :nan)
      true
      iex> Matrex.contains?(m, 9)
      false
  """
  @spec contains?(matrex, element) :: boolean
  def contains?(%Matrex{} = matrex, value), do: find(matrex, value) != nil

  @doc """
  Divides two matrices element-wise or matrix by scalar or scalar by matrix. NIF through `find/2`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[10, 20, 25], [8, 9, 4]])
      ...> |> Matrex.divide(Matrex.new([[5, 10, 5], [4, 3, 4]]))
      #Matrex[2×3]
      ┌                         ┐
      │     2.0     2.0     5.0 │
      │     2.0     3.0     1.0 │
      └                         ┘

      iex> Matrex.new([[10, 20, 25], [8, 9, 4]])
      ...> |> Matrex.divide(2)
      #Matrex[2×3]
      ┌                         ┐
      │     5.0    10.0    12.5 │
      │     4.0     4.5     2.0 │
      └                         ┘

      iex> Matrex.divide(100, Matrex.new([[10, 20, 25], [8, 16, 4]]))
      #Matrex[2×3]
      ┌                         ┐
      │    10.0     5.0     4.0 │
      │    12.5    6.25    25.0 │
      └                         ┘

  """
  @spec divide(matrex, matrex) :: matrex
  @spec divide(matrex, number) :: matrex
  @spec divide(number, matrex) :: matrex
  def divide(%Matrex{data: dividend} = _dividend, %Matrex{data: divisor} = _divisor),
    do: %Matrex{data: NIFs.divide(dividend, divisor)}

  def divide(%Matrex{data: matrix}, scalar) when is_number(scalar),
    do: %Matrex{data: NIFs.divide_by_scalar(matrix, scalar)}

  def divide(scalar, %Matrex{data: matrix}) when is_number(scalar),
    do: %Matrex{data: NIFs.divide_scalar(scalar, matrix)}

  @doc """
  Matrix multiplication. NIF, via `cblas_sgemm()`.

  Number of columns of the first matrix must be equal to the number of rows of the second matrix.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot(Matrex.new([[1, 2], [3, 4], [5, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot(matrex, matrex) :: matrex
  def dot(
        matrex_data(_rows1, columns1, _data1, first),
        matrex_data(rows2, _columns2, _data2, second)
      )
      when columns1 == rows2,
      do: %Matrex{data: NIFs.dot(first, second)}

  @doc """
  Matrix multiplication with addition of third matrix.  NIF, via `cblas_sgemm()`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_and_add(Matrex.new([[1, 2], [3, 4], [5, 6]]), Matrex.new([[1, 2], [3, 4]]))
      #Matrex[2×2]
      ┌                 ┐
      │    23.0    30.0 │
      │    52.0    68.0 │
      └                 ┘

  """
  @spec dot_and_add(matrex, matrex, matrex) :: matrex
  def dot_and_add(
        matrex_data(_rows1, columns1, _data1, first),
        matrex_data(rows2, _columns2, _data2, second),
        %Matrex{data: third}
      )
      when columns1 == rows2,
      do: %Matrex{data: NIFs.dot_and_add(first, second, third)}

  @doc """
  Computes dot product of two matrices, then applies math function to each element
  of the resulting matrix.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_and_apply(Matrex.new([[1, 2], [3, 4], [5, 6]]), :sqrt)
      #Matrex[2×2]
      ┌                 ┐
      │ 4.69042  5.2915 │
      │     7.0     8.0 │
      └                 ┘
  """
  @spec dot_and_apply(matrex, matrex, atom) :: matrex
  def dot_and_apply(
        matrex_data(_rows1, columns1, _data1, first),
        matrex_data(rows2, _columns2, _data2, second),
        function
      )
      when columns1 == rows2 and function in @math_functions,
      do: %Matrex{data: NIFs.dot_and_apply(first, second, function)}

  @doc """
  Matrix multiplication where the second matrix needs to be transposed.  NIF, via `cblas_sgemm()`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_nt(Matrex.new([[1, 3, 5], [2, 4, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot_nt(matrex, matrex) :: matrex
  def dot_nt(
        matrex_data(_rows1, columns1, _data1, first),
        matrex_data(_rows2, columns2, _data2, second)
      )
      when columns1 == columns2,
      do: %Matrex{data: NIFs.dot_nt(first, second)}

  @doc """
  Matrix dot multiplication where the first matrix needs to be transposed.  NIF, via `cblas_sgemm()`.

  The result is multiplied by scalar `alpha`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 4], [2, 5], [3, 6]]) |>
      ...> Matrex.dot_tn(Matrex.new([[1, 2], [3, 4], [5, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot_tn(matrex, matrex, number) :: matrex
  def dot_tn(
        matrex_data(rows1, _columns1, _data1, first),
        matrex_data(rows2, _columns2, _data2, second),
        alpha \\ 1.0
      )
      when rows1 == rows2 and is_number(alpha),
      do: %Matrex{data: NIFs.dot_tn(first, second, alpha)}

  @doc """
  Matrix cholesky decompose. NIF, via naive implementation.

  The first matrix must be symmetric and positive definitive.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[3, 4, 3], [4, 8, 6], [3, 6, 9]]) |>
      ...> Matrex.cholesky()
      #Matrex[3×3]
      ┌                         ┐
      │ 1.73205     0.0     0.0 │
      │  2.3094 1.63299     0.0 │
      │ 1.73205 1.22474 2.12132 │
      └                         ┘

  """
  @spec cholesky(matrex) :: matrex
  def cholesky(matrex_data(rows1, columns1, _data1, first))
      when rows1 == columns1,
      do: %Matrex{data: NIFs.cholesky(first)}

  @doc """
  Matrix forward substitution. NIF, via naive C implementation.

  The first matrix must be square while the
  number of columns of the first matrix must
  equal the number of rows of the second.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.forward_substitute(
      ...>   Matrex.new([[3, 4], [4, 8]]) |> Matrex.cholesky(),
      ...>   Matrex.new([[1],[2]]))
      #Matrex[2×1]
      ┌         ┐
      │ 0.57735 │
      │ 0.40825 │
      └         ┘

  """
  @spec forward_substitute(matrex, matrex) :: matrex
  def forward_substitute(
        matrex_data(rows1, columns1, _data1, first),
        matrex_data(rows2, columns2, _data2, second)
      )
      when rows1 == columns1 and rows1 == rows2 and columns2 == 1,
      do: %Matrex{data: NIFs.forward_substitute(first, second)}

  @doc """
  Create eye (identity) square matrix of given size.

  ## Examples

      iex> Matrex.eye(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     0.0     0.0 │
      │     0.0     1.0     0.0 │
      │     0.0     0.0     1.0 │
      └                         ┘

      iex> Matrex.eye(3, 2.95)
      #Matrex[3×3]
      ┌                         ┐
      │    2.95     0.0     0.0 │
      │     0.0    2.95     0.0 │
      │     0.0     0.0    2.95 │
      └                         ┘
  """
  @spec eye(index, element) :: matrex
  def eye(size, value \\ 1.0) when is_integer(size) and is_number(value),
    do: %Matrex{data: NIFs.eye(size, value)}

  @doc """
  Create new matrix with only diagonal elements from a given matrix.

  ## Examples

      iex> Matrex.eye(3) |> Matrex.diagonal()
      ┌                         ┐
      │    1.0      1.0     1.0 │
      └                         ┘

  """
  @spec diagonal(matrex) :: matrex
  def diagonal(matrix),
    do: %Matrex{data: NIFs.diagonal(matrix.data)}

  @doc """
  Create matrix filled with given value. NIF.

  ## Example

      iex> Matrex.fill(4,3, 55)
      #Matrex[4×3]
      ┌                         ┐
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      └                         ┘
  """
  @spec fill(index, index, element) :: matrex
  def fill(rows, cols, value)
      when (is_integer(rows) and is_integer(cols) and is_number(value)) or is_atom(value),
      do: %Matrex{data: NIFs.fill(rows, cols, float_to_binary(value))}

  @doc """
  Create square matrix filled with given value. Inlined.

  ## Example

      iex> Matrex.fill(3, 55)
      #Matrex[3×3]
      ┌                         ┐
      │    33.0    33.0    33.0 │
      │    33.0    33.0    33.0 │
      │    33.0    33.0    33.0 │
      └                         ┘
  """
  @spec fill(index, element) :: matrex
  def fill(size, value), do: fill(size, size, value)

  @doc """
  Find position of the first occurence of the given value in the matrix. NIF.

  Returns {row, column} tuple or nil, if nothing was found. One-based.

  ## Example


  """
  @spec find(matrex, element) :: {index, index} | nil
  def find(%Matrex{data: data}, value) when is_number(value) or value in [:nan, :inf, :neg_inf],
    do: NIFs.find(data, float_to_binary(value))

  @doc """
  Return first element of a matrix.

  ## Example

      iex> Matrex.new([[6,5,4],[3,2,1]]) |> Matrex.first()
      6.0

  """
  @spec first(matrex) :: element
  def first(matrex_data(_rows, _columns, <<element::binary-@element_size, _::binary>>)),
    do: binary_to_float(element)

  @doc """
  Prints monochrome or color heatmap of the matrix to the console.

  Supports 8, 256 and 16mln of colors terminals. Monochrome on 256 color palette is the default.

  `type` can be `:mono8`, `:color8`, `:mono256`, `:color256`, `:mono24bit` and `:color24bit`.

  Special float values, like infinity and not-a-number are marked with contrast colors on the map.

  ## Options

    * `:at` — positions heatmap at the specified `{row, col}` position inside terminal.
    * `:title` — sets the title of the heatmap.

  ## Examples

  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/mnist8.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/mnist_sum.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/magic_square.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/twin_peaks.png" width="220px"  />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/neurons_mono.png" width="233px"  />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/logistic_regression.gif" width="180px" />&nbsp;

  """
  @spec heatmap(
          matrex,
          :mono8 | :color8 | :mono256 | :color256 | :mono24bit | :color24bit,
          keyword
        ) :: matrex
  defdelegate heatmap(matrex, type \\ :mono256, opts \\ []), to: Matrex.Inspect

  @doc """
  An alias for `eye/1`.
  """
  @spec identity(index) :: matrex
  defdelegate identity(size), to: __MODULE__, as: :eye

  @doc """
  Returns list of all rows of a matrix as single-row matrices.

  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[6×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘
      iex> Matrex.list_of_rows(m)
      [#Matrex[1×2]
      ┌                 ┐
      │     1.0     2.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     3.0     4.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     5.0     6.0 │
      └                 ┘]


  """
  @spec list_of_rows(matrex) :: [matrex]
  def list_of_rows(matrex_data(rows, columns, matrix)) do
    do_list_rows(matrix, rows, columns)
  end

  @doc """
  Returns range of rows of a matrix as list of 1-row matrices.

  ## Example

      iex> m = Matrex.reshape(1..12, 6, 2)
      #Matrex[6×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      │     7.0     8.0 │
      │     9.0    10.0 │
      │    11.0    12.0 │
      └                 ┘
      iex> Matrex.list_of_rows(m, 2..4)
      [#Matrex[1×2]
      ┌                 ┐
      │     3.0     4.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     5.0     6.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     7.0     8.0 │
      └                 ┘]

  """
  @spec list_of_rows(matrex, Range.t()) :: [matrex]
  def list_of_rows(matrex_data(rows, columns, matrix), from..to)
      when from <= to and to <= rows do
    part =
      binary_part(
        matrix,
        (from - 1) * columns * @element_size,
        (to - from + 1) * columns * @element_size
      )

    do_list_rows(part, to - from + 1, columns)
  end

  defp do_list_rows(<<>>, 0, _), do: []

  defp do_list_rows(<<rows::binary>>, row_num, columns) do
    [
      matrex_data(1, columns, binary_part(rows, 0, columns * @element_size))
      | do_list_rows(
          binary_part(rows, columns * @element_size, (row_num - 1) * columns * @element_size),
          row_num - 1,
          columns
        )
    ]
  end

  @doc """
  Load matrex from file.

  .csv and .mtx (binary) formats are supported.

  ## Example

      iex> Matrex.load("test/matrex.csv")
      #Matrex[5×4]
      ┌                                 ┐
      │     0.0  4.8e-4-0.00517-0.01552 │
      │-0.01616-0.01622 -0.0161-0.00574 │
      │  6.8e-4     0.0     0.0     0.0 │
      │     0.0     0.0     0.0     0.0 │
      │     0.0     0.0     0.0     0.0 │
      └                                 ┘
  """
  @spec load(binary) :: matrex
  def load(file_name) when is_binary(file_name) do
    cond do
      :filename.extension(file_name) == ".gz" ->
        File.read!(file_name)
        |> :zlib.gunzip()
        |> do_load(String.split(file_name, ".") |> Enum.at(-2) |> String.to_existing_atom())

      :filename.extension(file_name) == ".csv" ->
        do_load(File.read!(file_name), :csv)

      :filename.extension(file_name) == ".mtx" ->
        do_load(File.read!(file_name), :mtx)

      :filename.extension(file_name) == ".idx" ->
        do_load(File.read!(file_name), :idx)

      true ->
        raise "Unknown file format: #{file_name}"
    end
  end

  @spec load(binary, :idx | :csv | :mtx) :: matrex
  def load(file_name, format) when format in [:idx, :mtx, :csv],
    do: do_load(File.read!(file_name), format)

  defp do_load(data, :csv), do: new(data)
  defp do_load(data, :mtx), do: %Matrex{data: data}
  defp do_load(data, :idx), do: %Matrex{data: Matrex.IDX.load(data)}

  @doc """
  Creates "magic" n*n matrix, where sums of all dimensions are equal.


  ## Example

      iex> Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
  """
  @spec magic(index) :: matrex
  def magic(n) when is_integer(n), do: Matrex.MagicSquare.new(n) |> new()

  @doc false
  # Shortcut to get functions list outside in Matrex.Operators module.
  def math_functions_list(), do: @math_functions

  @doc """
  Maximum element in a matrix. NIF.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.max(m)
      25.0

      iex> Matrex.reshape([1, 2, :inf, 4, 5, 6], 2, 3) |> max()
      :inf

  """
  @spec max(matrex) :: element
  def max(%Matrex{data: matrix}), do: NIFs.max(matrix)

  @doc """
  Returns maximum finite element of a matrex. NIF.

  Used on matrices which may contain infinite values.

  ## Example

      iex>Matrex.reshape([1, 2, :inf, 3, :nan, 5], 3, 2) |> Matrex.max_finite()
      5.0

  """
  @spec max_finite(matrex) :: float
  def max_finite(%Matrex{data: matrix}), do: NIFs.max_finite(matrix)

  @doc """

  Minimum element in a matrix. NIF.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.min(m)
      1.0

      iex> Matrex.reshape([1, 2, :neg_inf, 4, 5, 6], 2, 3) |> max()
      :neg_inf

  """
  @spec min(matrex) :: element
  def min(%Matrex{data: matrix}), do: NIFs.min(matrix)

  @doc """
  Returns minimum finite element of a matrex. NIF.

  Used on matrices which may contain infinite values.

  ## Example

      iex>Matrex.reshape([1, 2, :neg_inf, 3, 4, 5], 3, 2) |> Matrex.min_finite()
      1.0

  """
  @spec min_finite(matrex) :: float
  def min_finite(%Matrex{data: matrix}), do: NIFs.min_finite(matrix)

  @doc """
  Elementwise multiplication of two matrices or matrix and a scalar. NIF.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.multiply(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     5.0     4.0     3.0 │
      │    12.0    20.0    36.0 │
      └                         ┘

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |> Matrex.multiply(2)
      #Matrex[2×3]
      ┌                         ┐
      │     2.0     4.0     6.0 │
      │     8.0    10.0    12.0 │
      └                         ┘

  """
  @spec multiply(matrex, matrex) :: matrex
  @spec multiply(matrex, number) :: matrex
  @spec multiply(number, matrex) :: matrex
  def multiply(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.multiply(first, second)}

  def multiply(%Matrex{data: matrix}, scalar) when is_number(scalar),
    do: %Matrex{data: NIFs.multiply_with_scalar(matrix, scalar)}

  def multiply(scalar, %Matrex{data: matrix}) when is_number(scalar),
    do: %Matrex{data: NIFs.multiply_with_scalar(matrix, scalar)}

  @doc """
  Negates each element of the matrix. NIF.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |> Matrex.neg()
      #Matrex[2×3]
      ┌                         ┐
      │    -1.0    -2.0    -3.0 │
      │    -4.0    -5.0    -6.0 │
      └                         ┘

  """
  @spec neg(matrex) :: matrex
  def neg(%Matrex{data: matrix}), do: %Matrex{data: NIFs.neg(matrix)}

  @doc """
  Creates new matrix with values provided by the given function.

  If function accepts two arguments one-based row and column index of each element are passed to it.

  ## Examples

      iex> Matrex.new(3, 3, fn -> :rand.uniform() end)
      #Matrex[3×3]
      ┌                         ┐
      │ 0.45643 0.91533 0.25332 │
      │ 0.29095 0.21241  0.9776 │
      │ 0.42451 0.05422 0.92863 │
      └                         ┘

      iex> Matrex.new(3, 3, fn row, col -> row*col end)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     2.0     4.0     6.0 │
      │     3.0     6.0     9.0 │
      └                         ┘

  """
  @spec new(index, index, (() -> element)) :: matrex
  @spec new(index, index, (index, index -> element)) :: matrex
  def new(rows, columns, function) when is_function(function, 0) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    new_matrix_from_function(rows * columns, function, initial)
  end

  def new(rows, columns, function) when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>
    size = rows * columns

    new_matrix_from_function(size, rows, columns, function, initial)
  end

  @spec float_to_binary(element | :nan | :inf | :neg_inf) :: binary
  defp float_to_binary(val) when is_number(val), do: <<val::float-little-32>>
  defp float_to_binary(:nan), do: @not_a_number
  defp float_to_binary(:inf), do: @positive_infinity
  defp float_to_binary(:neg_inf), do: @negative_infinity

  defp float_to_binary(unknown_val),
    do: raise(ArgumentError, message: "Unknown matrix element value: #{unknown_val}")

  @doc """
  Creates new matrix from list of lists or text representation (compatible with MathLab/Octave).

  List of lists can contain other matrices, which are concatenated in one.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.new([[Matrex.fill(2, 1.0), Matrex.fill(2, 3, 2.0)],
      ...> [Matrex.fill(1, 2, 3.0), Matrex.fill(1, 3, 4.0)]])
      #Matrex[5×5]
      ┌                                         ┐
      │     1.0     1.0     2.0     2.0     2.0 │
      │     1.0     1.0     2.0     2.0     2.0 │
      │     3.0     3.0     4.0     4.0     4.0 │
      └                                         ┘

      iex> Matrex.new("1;0;1;0;1")
      #Matrex[5×1]
      ┌         ┐
      │     1.0 │
      │     0.0 │
      │     1.0 │
      │     0.0 │
      │     1.0 │
      └         ┘

      iex> Matrex.new(\"\"\"
      ...>         1.00000   0.10000   0.60000   1.10000
      ...>         1.00000   0.20000   0.70000   1.20000
      ...>         1.00000       NaN   0.80000   1.30000
      ...>             Inf   0.40000   0.90000   1.40000
      ...>         1.00000   0.50000    NegInf   1.50000
      ...>       \"\"\")
      #Matrex[5×4]
      ┌                                 ┐
      │     1.0     0.1     0.6     1.1 │
      │     1.0     0.2     0.7     1.2 │
      │     1.0    NaN      0.8     1.3 │
      │     ∞       0.4     0.9     1.4 │
      │     1.0     0.5    -∞       1.5 │
      └                                 ┘

  """
  @spec new([[element]] | [[matrex]] | binary) :: matrex
  def new(
        [
          [
            %Matrex{} | _
          ]
          | _
        ] = lol_of_ma
      ) do
    lol_of_ma
    |> Enum.map(&Matrex.concat/1)
    |> Enum.reduce(&Matrex.concat(&2, &1, :rows))
  end

  def new([first_list | _] = lol_or_binary) when is_list(first_list) do
    rows = length(lol_or_binary)
    columns = length(first_list)

    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    %Matrex{
      data:
        Enum.reduce(lol_or_binary, initial, fn list, accumulator ->
          accumulator <>
            Enum.reduce(list, <<>>, fn element, partial ->
              <<partial::binary, float_to_binary(element)::binary>>
            end)
        end)
    }
  end

  def new(text) when is_binary(text) do
    text
    |> String.split(["\n", ";"], trim: true)
    |> Enum.map(fn line ->
      line
      |> String.split(["\s", ","], trim: true)
      |> Enum.map(fn f -> parse_float(f) end)
    end)
    |> new()
  end

  @spec parse_float(binary) :: element | :nan | :inf | :neg_inf
  defp parse_float("NaN"), do: :nan
  defp parse_float("Inf"), do: :inf
  defp parse_float("+Inf"), do: :inf
  defp parse_float("-Inf"), do: :neg_inf
  defp parse_float("NegInf"), do: :neg_inf

  defp parse_float(string) do
    case Float.parse(string) do
      {value, _rem} -> value
      :error -> raise ArgumentError, message: "Unparseable matrix element value: #{string}"
    end
  end

  defp new_matrix_from_function(0, _, accumulator), do: %Matrex{data: accumulator}

  defp new_matrix_from_function(size, function, accumulator),
    do:
      new_matrix_from_function(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-32>>
      )

  defp new_matrix_from_function(0, _, _, _, accumulator), do: %Matrex{data: accumulator}

  defp new_matrix_from_function(size, rows, columns, function, accumulator) do
    {row, col} =
      if rem(size, columns) == 0 do
        {rows - div(size, columns), 0}
      else
        {rows - 1 - div(size, columns), columns - rem(size, columns)}
      end

    new_accumulator = <<accumulator::binary, function.(row + 1, col + 1)::float-little-32>>

    new_matrix_from_function(size - 1, rows, columns, function, new_accumulator)
  end

  @doc """
  Bring all values of matrix into [0, 1] range. NIF.

  Where 0 corresponds to the minimum value of the matrix, and 1 — to the maxixmim.

  ## Example

      iex> m = Matrex.reshape(1..9, 3, 3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      └                         ┘
      iex> Matrex.normalize(m)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0   0.125    0.25 │
      │   0.375     0.5   0.625 │
      │    0.75   0.875     1.0 │
      └                         ┘
  """
  @spec normalize(matrex) :: matrex
  def normalize(%Matrex{data: data}), do: %Matrex{data: NIFs.normalize(data)}

  @doc """
  Create matrix filled with ones.

  ## Example

      iex> Matrex.ones(2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(index, index) :: matrex
  def ones(rows, cols) when is_integer(rows) and is_integer(cols), do: fill(rows, cols, 1)

  @doc """
  Create matrex of ones of square dimensions or consuming output of `size/1` function.

  ## Examples

      iex> Matrex.ones(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘

      iex> m = Matrex.new("1 2 3; 4 5 6")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.ones(Matrex.size(m))
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(index) :: matrex
  @spec ones({index, index}) :: matrex
  def ones({rows, cols}), do: ones(rows, cols)

  def ones(size) when is_integer(size), do: fill(size, 1)

  @doc """
  Prints matrix to the console.

  Accepted options:
    * `:rows` — number of rows of matrix to show. Defaults to 21
    * `:columns` — number of columns of matrix to show. Defaults to maximum number of column,
    that fits into current terminal width.

    Returns the matrix itself, so can be used in pipes.

    ## Example

        iex> print(m, rows: 5, columns: 3)
        #Matrex[20×20]
        ┌                             ┐
        │     1.0   399.0  …     20.0 │
        │   380.0    22.0  …    361.0 │
        │   360.0    42.0  …    341.0 │
        │     ⋮       ⋮     …      ⋮  │
        │    40.0   362.0  …     21.0 │
        │   381.0    19.0  …    400.0 │
        └                             ┘

  """
  @spec print(matrex, Keyword.t()) :: matrex
  def print(%Matrex{} = matrex, opts \\ [rows: 21]) do
    {:ok, terminal_columns} = :io.columns()

    columns =
      case Keyword.get(opts, :columns) do
        nil -> terminal_columns
        cols -> cols * 8 + 10
      end

    matrex
    |> Matrex.Inspect.do_inspect(columns, Keyword.get(opts, :rows, 21))
    |> IO.puts()

    matrex
  end

  @doc """
  Create matrix of random floats in [0, 1] range. NIF.

  C language RNG is seeded on NIF libray load with `srandom(time(NULL) + clock())`.

  ## Example

      iex> Matrex.random(4,3)
      #Matrex[4×3]
      ┌                         ┐
      │ 0.32994 0.28736 0.88012 │
      │ 0.51782 0.68608 0.29976 │
      │ 0.52953  0.9071 0.26743 │
      │ 0.82189 0.59311  0.8451 │
      └                         ┘

  """
  @spec random(index, index) :: matrex
  def random(rows, columns) when is_integer(rows) and is_integer(columns),
    do: %Matrex{data: NIFs.random(rows, columns)}

  @doc """
  Create square matrix of random floats.

  See `random/2` for details.

  ## Example

      iex> Matrex.random(3)
      #Matrex[3×3]
      ┌                         ┐
      │ 0.66438 0.31026 0.98602 │
      │ 0.82127 0.04701 0.13278 │
      │ 0.96935 0.70772 0.98738 │
      └                         ┘
  """
  @spec random(index) :: matrex
  def random(size) when is_integer(size), do: random(size, size)

  @doc """
  Resize matrix by scaling its dimenson with `scale`. NIF.

  ## Examples

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex(3)> Matrex.resize(m, 2)
      #Matrex[6×6]
      ┌                                                 ┐
      │     8.0     8.0     1.0     1.0     6.0     6.0 │
      │     8.0     8.0     1.0     1.0     6.0     6.0 │
      │     3.0     3.0     5.0     5.0     7.0     7.0 │
      │     3.0     3.0     5.0     5.0     7.0     7.0 │
      │     4.0     4.0     9.0     9.0     2.0     2.0 │
      │     4.0     4.0     9.0     9.0     2.0     2.0 │
      └                                                 ┘

      iex(4)> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex(5)> Matrex.resize(m, 0.5)
      #Matrex[3×3]
      ┌                         ┐
      │    16.0    23.0     7.0 │
      │    22.0     4.0    13.0 │
      │     9.0    11.0    25.0 │
      └                         ┘
  """
  @spec resize(matrex, number, :nearest | :bilinear) :: matrex
  def resize(matrex, scale, method \\ :nearest)

  def resize(%Matrex{} = matrex, 1, _), do: matrex

  def resize(%Matrex{data: data}, scale, :nearest) when is_number(scale) and scale > 0,
    do: %Matrex{data: NIFs.resize(data, scale)}

  @doc """
  Reshapes list of values into a matrix of given size or changes the shape of existing matrix.

  Takes a list or anything, that implements `Enumerable.to_list/1`.

  Can take a list of matrices and concatenate them into one big matrix.

  Raises `ArgumentError` if list size and given shape do not match.

  ## Example

      iex> [1, 2, 3, 4, 5, 6] |> Matrex.reshape(2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.reshape([Matrex.zeros(2), Matrex.ones(2),
      ...> Matrex.fill(3, 2, 2.0), Matrex.fill(3, 2, 3.0)], 2, 2)
      #Matrex[5×4]
      ┌                                 ┐
      │     0.0     0.0     1.0     1.0 │
      │     0.0     0.0     1.0     1.0 │
      │     2.0     2.0     3.0     3.0 │
      │     2.0     2.0     3.0     3.0 │
      │     2.0     2.0     3.0     3.0 │
      └                                 ┘

      iex> Matrex.reshape(1..6, 2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.new("1 2 3; 4 5 6") |> Matrex.reshape(3, 2)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘

  """
  def reshape([], _, _), do: raise(ArgumentError)

  @spec reshape([matrex], index, index) :: matrex
  def reshape([%Matrex{} | _] = enum, _rows, columns) do
    enum
    |> Enum.chunk_every(columns)
    |> new()
  end

  @spec reshape([element], index, index) :: matrex
  def reshape([_ | _] = list, rows, columns),
    do: %Matrex{
      data:
        do_reshape(
          <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>,
          list,
          rows,
          columns
        )
    }

  @spec reshape(matrex, index, index) :: matrex
  def reshape(
        matrex_data(rows, columns, _matrix),
        new_rows,
        new_columns
      )
      when rows * columns != new_rows * new_columns,
      do:
        raise(
          ArgumentError,
          message:
            "Cannot reshape: #{rows}×#{columns} does not fit into #{new_rows}×#{new_columns}."
        )

  def reshape(
        matrex_data(rows, columns, _matrix) = matrex,
        rows,
        columns
      ),
      # No need to reshape.
      do: matrex

  def reshape(
        matrex_data(_rows, _columns, matrix),
        new_rows,
        new_columns
      ),
      do: matrex_data(new_rows, new_columns, matrix)

  @spec reshape(Range.t(), index, index) :: matrex
  def reshape(a..b, rows, cols) when b - a + 1 != rows * cols,
    do:
      raise(
        ArgumentError,
        message: "range #{a}..#{b} cannot be reshaped into #{rows}×#{cols} matrix."
      )

  def reshape(a..b, rows, cols), do: %Matrex{data: NIFs.from_range(a, b, rows, cols)}

  @spec reshape(Enumerable.t(), index, index) :: matrex
  def reshape(input, rows, columns), do: input |> Enum.to_list() |> reshape(rows, columns)

  defp do_reshape(data, [], 1, 0), do: data

  defp do_reshape(_data, [], _, _),
    do: raise(ArgumentError, message: "Not enough elements for this shape")

  defp do_reshape(_data, [_ | _], 1, 0),
    do: raise(ArgumentError, message: "Too much elements for this shape")

  # Another row is ready, restart counters
  defp do_reshape(
         <<_rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, _::binary>> =
           data,
         list,
         row,
         0
       ),
       do: do_reshape(data, list, row - 1, columns)

  defp do_reshape(
         <<_rows::unsigned-integer-little-32, _columns::unsigned-integer-little-32, _::binary>> =
           data,
         [elem | tail],
         row,
         column
       ) do
    do_reshape(<<data::binary, float_to_binary(elem)::binary-4>>, tail, row, column - 1)
  end

  @doc """
  Return matrix row as list by one-based index.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.row_to_list(m, 3)
      [3.0, 10.0, 12.0, 19.0, 21.0]
  """
  @spec row_to_list(matrex, index) :: [element]
  def row_to_list(%Matrex{data: matrix}, row) when is_integer(row) and row > 0,
    do: NIFs.row_to_list(matrix, row - 1)

  @doc """
  Get row of matrix as matrix (vector) in matrex form. One-based.

  You can use shorter `matrex[n]` syntax for the same result.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.row(m, 4)
      #Matrex[1×5]
      ┌                                         ┐
      │     9.0    11.0    18.0    25.0     2.0 │
      └                                         ┘
      iex> m[4]
      #Matrex[1×5]
      ┌                                         ┐
      │     9.0    11.0    18.0    25.0     2.0 │
      └                                         ┘
  """
  @spec row(matrex, index) :: matrex
  def row(matrex_data(rows, columns, data), row)
      when is_integer(row) and row > 0 and row <= rows do
    matrex_data(
      1,
      columns,
      binary_part(data, (row - 1) * columns * @element_size, columns * @element_size)
    )
  end

  @doc """
  Saves matrex into file.

  Binary (.mtx) and CSV formats are supported currently.

  Format is defined by the extension of the filename.

  ## Example

      iex> Matrex.random(5) |> Matrex.save("r.mtx")
      :ok
  """
  @spec save(matrex, binary) :: :ok | :error
  def save(
        %Matrex{
          data: matrix
        },
        file_name
      )
      when is_binary(file_name) do
    cond do
      :filename.extension(file_name) == ".mtx" ->
        File.write!(file_name, matrix)

      :filename.extension(file_name) == ".csv" ->
        csv =
          matrix
          |> NIFs.to_list_of_lists()
          |> Enum.reduce("", fn row_list, acc ->
            acc <>
              Enum.reduce(row_list, "", fn elem, line ->
                line <> element_to_string(elem) <> ","
              end) <> "\n"
          end)

        File.write!(file_name, csv)

      true ->
        raise "Unknown file format: #{file_name}"
    end
  end

  @doc false
  @spec element_to_string(element) :: binary
  # Save zero values without fraction part to save space
  def element_to_string(0.0), do: "0"
  def element_to_string(val) when is_float(val), do: Float.to_string(val)
  def element_to_string(:nan), do: "NaN"
  def element_to_string(:inf), do: "Inf"
  def element_to_string(:neg_inf), do: "-Inf"

  @doc """
  Transfer one-element matrix to a scalar value.

  Differently from `first/1` will not match and throw an error,
  if matrix contains more than one element.

  ## Example

      iex> Matrex.new([[1.234]]) |> Matrex.scalar()
      1.234

      iex> Matrex.new([[0]]) |> Matrex.divide(0) |> Matrex.scalar()
      :nan

      iex> Matrex.new([[1.234, 5.678]]) |> Matrex.scalar()
      ** (FunctionClauseError) no function clause matching in Matrex.scalar/1
  """
  @spec scalar(matrex) :: element
  def scalar(%Matrex{
        data: <<1::unsigned-integer-little-32, 1::unsigned-integer-little-32, elem::binary-4>>
      }),
      do: binary_to_float(elem)

  @doc """
  Set element of matrix at the specified position (one-based) to new value.

  ## Example

      iex> m = Matrex.ones(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
      iex> m = Matrex.set(m, 2, 2, 0)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
      iex> m = Matrex.set(m, 3, 2, :neg_inf)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     -∞      1.0 │
      └                         ┘
  """
  @spec set(matrex, index, index, element) :: matrex
  def set(matrex_data(rows, cols, _rest, matrix), row, column, value)
      when (is_number(value) or value in [:nan, :inf, :neg_inf]) and row > 0 and column > 0 and
             row <= rows and column <= cols,
      do: %Matrex{data: NIFs.set(matrix, row - 1, column - 1, float_to_binary(value))}

  @doc """
  Set column of a matrix to the values from the given 1-column matrix. NIF.

  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[3×2]
      ┌                    ┐
      │     1.0     2.0    │
      │     3.0     4.0    │
      │     5.0     6.0    │
      └                    ┘

      iex> Matrex.set_column(m, 2, Matrex.new("7; 8; 9"))
      #Matrex[3×2]
      ┌                    ┐
      │     1.0     7.0    │
      │     3.0     8.0    │
      │     5.0     9.0    │
      └                    ┘
  """
  @spec set_column(matrex, index, matrex) :: matrex
  def set_column(
        matrex_data(rows, columns, _rest1, matrix),
        column,
        matrex_data(rows, 1, _rest2, column_matrix)
      )
      when column in 1..columns,
      do: %Matrex{data: NIFs.set_column(matrix, column - 1, column_matrix)}

  @doc """
  Return size of matrix as `{rows, cols}`

  ## Example

      iex> m = Matrex.random(2,3)
      #Matrex[2×3]
      ┌                         ┐
      │ 0.69745 0.23668 0.36376 │
      │ 0.63423 0.29651 0.22844 │
      └                         ┘
      iex> Matrex.size(m)
      {2, 3}
  """
  @spec size(matrex) :: {index, index}
  def size(matrex_data(rows, cols, _)), do: {rows, cols}

  @doc """
  Produces element-wise squared matrix. NIF through `multiply/4`.


  ## Example

      iex> m = Matrex.new("1 2 3; 4 5 6")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.square(m)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     4.0     9.0 │
      │    16.0    25.0    36.0 │
      └                         ┘

  """
  @spec square(matrex) :: matrex
  def square(%Matrex{data: matrix}), do: %Matrex{data: Matrex.NIFs.multiply(matrix, matrix)}

  @doc """
  Returns submatrix for a given matrix. NIF.

  Rows and columns ranges are inclusive and one-based.

  ## Example

      iex> m = Matrex.new("1 2 3; 4 5 6; 7 8 9")
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      └                         ┘
      iex> Matrex.submatrix(m, 2..3, 2..3)
      #Matrex[2×2]
      ┌                ┐
      │    5.0     6.0 │
      │    8.0     9.0 │
      └                ┘
  """
  @spec submatrix(matrex, Range.t(), Range.t()) :: matrex
  def submatrix(matrex_data(rows, cols, _rest, data), row_from..row_to, col_from..col_to)
      when row_from in 1..rows and row_to in row_from..rows and col_from in 1..cols and
             col_to in col_from..cols,
      do: %Matrex{data: NIFs.submatrix(data, row_from - 1, row_to - 1, col_from - 1, col_to - 1)}

  def submatrix(%Matrex{} = matrex, rows, cols) do
    raise(
      RuntimeError,
      "Submatrix position out of range or malformed: position is " <>
        "(#{Kernel.inspect(rows)}, #{Kernel.inspect(cols)}), source size is " <>
        "(#{Kernel.inspect(1..matrex[:rows])}, #{Kernel.inspect(1..matrex[:columns])})"
    )
  end

  @doc """
  Subtracts two matrices or matrix from scalar element-wise. NIF.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.subtract(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │    -4.0     0.0     2.0 │
      │     1.0     1.0     0.0 │
      └                         ┘

      iex> Matrex.subtract(1, Matrex.new([[1, 2, 3], [4, 5, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     0.0     -1.0   -2.0 │
      │    -3.0    -4.0    -5.0 │
      └                         ┘
  """
  @spec subtract(matrex | number, matrex | number) :: matrex
  def subtract(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.subtract(first, second)}

  def subtract(scalar, %Matrex{data: matrix}) when is_number(scalar),
    do: %Matrex{data: NIFs.subtract_from_scalar(scalar, matrix)}

  def subtract(%Matrex{data: matrix}, scalar) when is_number(scalar),
    do: %Matrex{data: NIFs.add_scalar(matrix, -scalar)}

  @doc """
  Subtracts the second matrix or scalar from the first. Inlined.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.subtract_inverse(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     4.0     0.0    -2.0 │
      │    -1.0    -1.0     0.0 │
      └                         ┘

      iex> Matrex.eye(3) |> Matrex.subtract_inverse(1)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     1.0     0.0 │
      └                         ┘

  """
  @spec subtract_inverse(matrex | number, matrex | number) :: matrex
  def subtract_inverse(%Matrex{} = first, %Matrex{} = second), do: subtract(second, first)

  def subtract_inverse(%Matrex{} = first, scalar) when is_number(scalar),
    do: subtract(scalar, first)

  def subtract_inverse(scalar, %Matrex{} = second) when is_number(scalar),
    do: subtract(second, scalar)

  @doc """
  Sums all elements. NIF.

  Can return special float values as atoms.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.sum(m)
      45.0

      iex> m = Matrex.new("1 Inf; 2 3")
      #Matrex[2×2]
      ┌                 ┐
      │     1.0     ∞   │
      │     2.0     3.0 │
      └                 ┘
      iex> sum(m)
      :inf
  """
  @spec sum(matrex) :: element
  def sum(%Matrex{data: matrix}), do: NIFs.sum(matrix)

  @doc """
  Trace of matrix (sum of all diagonal elements). Elixir.

  Can return special float values as atoms.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.trace(m)
      15.0

      iex> m = Matrex.new("Inf 1; 2 3")
      #Matrex[2×2]
      ┌                 ┐
      │     ∞       1.0 │
      │     2.0     3.0 │
      └                 ┘
      iex> trace(m)
      :inf
  """
  @spec trace(matrex) :: element
  def trace(%Matrex{data: matrix}), do: NIFs.diagonal(matrix) |> NIFs.sum()

  @doc """
  Converts to flat list. NIF.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_list(m)
      [8.0, 1.0, 6.0, 3.0, 5.0, 7.0, 4.0, 9.0, 2.0]
  """
  @spec to_list(matrex) :: list(element)
  def to_list(%Matrex{data: matrix}), do: NIFs.to_list(matrix)

  @doc """
  Converts to list of lists. NIF.

  ## Examples

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_list_of_lists(m)
      [[8.0, 1.0, 6.0], [3.0, 5.0, 7.0], [4.0, 9.0, 2.0]]

      iex> r = Matrex.divide(Matrex.eye(3), Matrex.zeros(3))
      #Matrex[3×3]
      ┌                         ┐
      │     ∞      NaN     NaN  │
      │    NaN      ∞      NaN  │
      │    NaN     NaN      ∞   │
      └                         ┘
      iex> Matrex.to_list_of_lists(r)
      [[:inf, :nan, :nan], [:nan, :inf, :nan], [:nan, :nan, :inf]]

  """
  @spec to_list_of_lists(matrex) :: list(list(element))
  def to_list_of_lists(%Matrex{data: matrix}), do: NIFs.to_list_of_lists(matrix)

  @doc """
  Convert any matrix m×n to a column matrix (m*n)×1.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_column(m)
      #Matrex[1×9]
      ┌                                                                         ┐
      │     8.0     1.0     6.0     3.0     5.0     7.0     4.0     9.0     2.0 │
      └                                                                         ┘

  """
  @spec to_column(matrex) :: matrex
  def to_column(matrex_data(_rows, 1, _rest) = m), do: m
  def to_column(matrex_data(rows, columns, _rest) = m), do: reshape(m, rows * columns, 1)

  @doc """
  Convert any matrix m×n to a row matrix 1×(m*n).

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_row(m)
      #Matrex[1×9]
      ┌                                                                         ┐
      │     8.0     1.0     6.0     3.0     5.0     7.0     4.0     9.0     2.0 │
      └                                                                         ┘

  """
  @spec to_row(matrex) :: matrex
  def to_row(matrex_data(1, _columns, _rest) = m), do: m
  def to_row(matrex_data(rows, columns, _rest) = m), do: reshape(m, 1, rows * columns)

  @doc """
  Transposes a matrix. NIF.

  ## Example

      iex> m = Matrex.new([[1,2,3],[4,5,6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.transpose(m)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     4.0 │
      │     2.0     5.0 │
      │     3.0     6.0 │
      └                 ┘
  """
  @spec transpose(matrex) :: matrex
  # Vectors are transposed by simply reshaping
  def transpose(matrex_data(1, columns, _rest) = m), do: reshape(m, columns, 1)
  def transpose(matrex_data(rows, 1, _rest) = m), do: reshape(m, 1, rows)
  def transpose(%Matrex{data: matrix}), do: %Matrex{data: NIFs.transpose(matrix)}

  @doc """
  Updates the element at the given position in matrix with function.

  Function is invoked with the current element value


  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘
      iex> Matrex.update(m, 2, 2, fn x -> x * x end)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0    16.0 │
      │     5.0     6.0 │
      └                 ┘

  """
  @spec update(matrex, index, index, (element -> element)) :: matrex
  def update(matrex_data(rows, columns, _data), row, col, _fun)
      when not inside_matrex(row, col, rows, columns),
      do:
        raise(
          ArgumentError,
          message: "Position (#{row}, #{col}) is out of matrex [#{rows}×#{columns}]"
        )

  def update(matrex_data(_rows, columns, data, matrix), row, col, fun)
      when is_function(fun, 1) do
    new_value =
      data
      |> binary_part(((row - 1) * columns + (col - 1)) * @element_size, @element_size)
      |> binary_to_float()
      |> fun.()
      |> float_to_binary()

    %Matrex{data: NIFs.set(matrix, row - 1, col - 1, new_value)}
  end

  @doc """
  Create matrix of zeros of the specified size. NIF, using `memset()`.

  Faster, than `fill(rows, cols, 0)`.

  ## Example

      iex> Matrex.zeros(4,3)
      #Matrex[4×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(index, index) :: matrex
  def zeros(rows, cols) when is_integer(rows) and is_integer(cols),
    do: %Matrex{data: NIFs.zeros(rows, cols)}

  @doc """
  Create square matrix of size `size` rows × `size` columns, filled with zeros. Inlined.

  ## Example

      iex> Matrex.zeros(3)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(index | {index, index}) :: matrex
  def zeros({rows, cols}), do: zeros(rows, cols)
  def zeros(size), do: zeros(size, size)
end
