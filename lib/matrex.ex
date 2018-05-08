defmodule Matrex do
  @moduledoc """
  Performs fast operations on matrices using native C code and CBLAS library.
  """

  alias Matrex.NIFs

  @enforce_keys [:data]
  defstruct [:data]
  @type t :: %Matrex{data: binary}

  @behaviour Access

  # Horizontal vector
  @impl Access
  def fetch(
        %Matrex{
          data:
            <<
              rows::unsigned-integer-little-32,
              _columns::unsigned-integer-little-32,
              _rest::binary
            >> = data
        },
        key
      )
      when is_integer(key) and rows == 1,
      do: {:ok, at(data, 0, key - 1)}

  # Vertical vector
  @impl Access
  def fetch(
        %Matrex{
          data:
            <<
              _rows::unsigned-integer-little-32,
              columns::unsigned-integer-little-32,
              _rest::binary
            >> = data
        },
        key
      )
      when is_integer(key) and columns == 1,
      do: {:ok, at(data, key - 1, 0)}

  @impl Access
  def fetch(
        %Matrex{
          data: data
        },
        key
      )
      when is_integer(key),
      do: {:ok, %Matrex{data: row(data, key - 1)}}

  @impl Access
  def fetch(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            _columns::unsigned-integer-little-32,
            _rest::binary
          >>
        },
        :rows
      ),
      do: {:ok, rows}

  @impl Access
  def fetch(
        %Matrex{
          data: <<
            _rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            _rest::binary
          >>
        },
        :cols
      ),
      do: {:ok, columns}

  @doc """
  Adds two matrices
  """
  def add(%Matrex{data: first}, %Matrex{data: second}), do: %Matrex{data: NIFs.add(first, second)}

  @doc """
  Apply C math function to matrix elementwise.
  """
  @spec apply(Matrex.t(), atom) :: Matrex.t()
  def apply(%Matrex{data: data} = matrix, function)
      when function in [
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
             :trunc,
             :round,
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
           ] do
    {rows, cols} = size(matrix)

    %Matrex{
      data:
        if(
          rows * cols < 100_000,
          do: NIFs.apply_math(data, function),
          else: NIFs.apply_parallel_math(data, function)
        )
    }
  end

  @doc """
  Applies the given function on each element of the matrix
  """
  @spec apply(Matrex.t(), function) :: Matrex.t()
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

  @spec apply(Matrex.t(), function) :: Matrex.t()
  def apply(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        function
      )
      when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>
    size = rows * columns

    %Matrex{data: apply_on_matrix(data, function, 1, size, initial)}
  end

  @spec apply(Matrex.t(), function) :: Matrex.t()
  def apply(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        function
      )
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
  """
  @spec apply(Matrex.t(), Matrex.t(), function) :: Matrex.t()
  def apply(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            first_data::binary
          >>
        },
        %Matrex{
          data: <<
            _::unsigned-integer-little-32,
            _::unsigned-integer-little-32,
            second_data::binary
          >>
        },
        function
      )
      when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    %Matrex{data: apply_on_matrices(first_data, second_data, function, initial)}
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
  Returns the index of the biggest element.
  """
  @spec argmax(Matrex.t()) :: non_neg_integer
  def argmax(%Matrex{data: data}), do: NIFs.argmax(data)

  @doc """
  Get element of a matrix at given zero-based position.
  """
  @spec at(Matrex.t(), non_neg_integer, non_neg_integer) :: float
  def at(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        row,
        col
      )
      when is_integer(row) and is_integer(col) do
    if row < 0 or row >= rows,
      do: raise(ArgumentError, message: "Row position out of range: #{row}")

    if col < 0 or col >= columns,
      do: raise(ArgumentError, message: "Column position out of range: #{col}")

    <<elem::float-little-32>> = binary_part(data, (row * columns + col) * 4, 4)
    elem
  end

  @doc """
  Get column of matrix as matrix (vector) in binary form.
  """
  @spec column(Matrex.t(), non_neg_integer) :: Matrex.t()
  def column(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        col
      )
      when is_integer(col) and col < columns do
    column = <<rows::unsigned-integer-little-32, 1::unsigned-integer-little-32>>

    %Matrex{
      data:
        0..(rows - 1)
        |> Enum.reduce(column, fn row, acc ->
          <<acc::binary, binary_part(data, (row * columns + col) * 4, 4)::binary>>
        end)
    }
  end

  @doc """
  Get column of matrix as list of floats
  """
  @spec column_as_list(Matrex.t(), non_neg_integer) :: list(float)
  def column_as_list(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        col
      )
      when is_integer(col) and col < columns do
    0..(rows - 1)
    |> Enum.map(fn row ->
      <<elem::float-little-32>> = binary_part(data, (row * columns + col) * 4, 4)
      elem
    end)
  end

  @doc """
  Divides two matrices
  """
  @spec divide(Matrex.t(), Matrex.t()) :: Matrex.t()
  def divide(%Matrex{data: dividend}, %Matrex{data: divisor}),
    do: %Matrex{data: NIFs.divide(dividend, divisor)}

  @doc """
  Matrix multiplication
  """
  @spec dot(Matrex.t(), Matrex.t()) :: Matrex.t()
  def dot(%Matrex{data: first}, %Matrex{data: second}), do: %Matrex{data: NIFs.dot(first, second)}

  @doc """
  Matrix multiplication with addition of thitd matrix
  """
  @spec dot_and_add(Matrex.t(), Matrex.t(), Matrex.t()) :: Matrex.t()
  def dot_and_add(%Matrex{data: first}, %Matrex{data: second}, %Matrex{data: third}),
    do: %Matrex{data: NIFs.dot_and_add(first, second, third)}

  @doc """
  Matrix multiplication where the second matrix needs to be transposed.
  """
  @spec dot_nt(Matrex.t(), Matrex.t()) :: Matrex.t()
  def dot_nt(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.dot_nt(first, second)}

  @doc """
  Matrix multiplication where the first matrix needs to be transposed.
  """
  @spec dot_tn(Matrex.t(), Matrex.t()) :: Matrex.t()
  def dot_tn(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.dot_tn(first, second)}

  @doc """
  Create eye square matrix of given size
  """
  @spec eye(non_neg_integer) :: Matrix.t()
  def eye(size) when is_integer(size), do: %Matrex{data: NIFs.eye(size)}

  @doc """
  Create matrix filled with given value
  """
  @spec fill(non_neg_integer, non_neg_integer, non_neg_integer) :: Matrex.t()
  def fill(rows, cols, value)
      when is_integer(rows) and is_integer(cols) and is_integer(value),
      do: %Matrex{data: NIFs.fill(rows, cols, value)}

  @doc """
  Create square matrix filled with given value
  """
  @spec fill(integer, integer) :: Matrex.t()
  def fill(size, value), do: fill(size, size, value)

  @doc """
  Return first element of a matrix.
  """
  @spec first(Matrex.t()) :: float
  def first(%Matrex{
        data: <<
          _rows::unsigned-integer-little-32,
          _columns::unsigned-integer-little-32,
          element::float-little-32,
          _rest::binary
        >>
      }),
      do: element

  @doc """
  Displays a visualization of the matrix.
  Set the second parameter to true to show full numbers.
  Otherwise, they are truncated.
  """
  @spec inspect(Matrex.t(), boolean) :: Matrex.t()
  def inspect(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            rest::binary
          >>
        } = matrex,
        full \\ false
      ) do
    IO.puts("Rows: #{rows} Columns: #{columns}")

    inspect_element(1, columns, rest, full)

    matrex
  end

  defp inspect_element(_, _, <<>>, _), do: :ok

  defp inspect_element(column, columns, <<element::float-little-32, rest::binary>>, full) do
    next_column =
      case column == columns do
        true ->
          IO.puts(undot(element, full))

          1.0

        false ->
          IO.write("#{undot(element, full)} ")

          column + 1.0
      end

    inspect_element(next_column, columns, rest, full)
  end

  defp undot(f, false) when is_float(f) and f - trunc(f) == 0.0, do: trunc(f)
  defp undot(f, false) when is_float(f), do: :io_lib.format("~7.3f", [f])
  defp undot(f, true) when is_float(f), do: f

  @doc """
  Creates "magic" n*n matrix, where sums of all dimensions are equal
  """
  @spec magic(non_neg_integer) :: Matrex.t()
  def magic(n) when is_integer(n), do: Matrex.MagicSquare.new(n) |> new()

  @doc """
  Maximum element in a matrix.
  """
  @spec max(Matrex.t()) :: float
  def max(%Matrex{data: matrix}), do: NIFs.max(matrix)

  @doc """
  Elementwise multiplication of two matrices
  """
  @spec multiply(Matrex.t(), Matrex.t()) :: Matrex.t()
  def multiply(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.multiply(first, second)}

  @doc """
  Elementwise multiplication of a scalar
  """
  @spec multiply_with_scalar(Matrex.t(), number) :: Matrex.t()
  def multiply_with_scalar(%Matrex{data: matrix}, scalar) when is_number(scalar),
    do: %Matrex{data: NIFs.multiply_with_scalar(matrix, scalar)}

  @doc """
  Creates a new matrix with values provided by the given function
  """
  @spec new(non_neg_integer, non_neg_integer, function) :: Matrex.t()
  def new(rows, columns, function) when is_function(function, 0) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    new_matrix_from_function(rows * columns, function, initial)
  end

  @spec new(non_neg_integer, non_neg_integer, function) :: Matrex.t()
  def new(rows, columns, function) when is_function(function, 2) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>
    size = rows * columns

    new_matrix_from_function(size, rows, columns, function, initial)
  end

  @spec new(non_neg_integer, non_neg_integer, list(list)) :: Matrex.t()
  def new(rows, columns, list_of_lists) when is_list(list_of_lists) do
    initial = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>

    %Matrex{
      data:
        Enum.reduce(list_of_lists, initial, fn list, accumulator ->
          accumulator <>
            Enum.reduce(list, <<>>, fn element, partial ->
              <<partial::binary, element::float-little-32>>
            end)
        end)
    }
  end

  @doc """
  Creates new matrix from list of lists.
  """
  @spec new(list(list)) :: Matrex.t()
  def new(list_of_lists) when is_list(list_of_lists) do
    rows = length(list_of_lists)
    cols = length(List.first(list_of_lists))
    new(rows, cols, list_of_lists)
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

    new_accumulator = <<accumulator::binary, function.(row, col)::float-little-32>>

    new_matrix_from_function(size - 1, rows, columns, function, new_accumulator)
  end

  @doc """
  Create matrix filled with ones.
  """
  @spec ones(non_neg_integer, non_neg_integer) :: Matrex.t()
  def ones(rows, cols) when is_integer(rows) and is_integer(cols), do: fill(rows, cols, 1)

  @doc """
  Create square matrix filled with ones.
  """
  @spec ones(non_neg_integer) :: Matrex.t()
  def ones(size) when is_integer(size), do: fill(size, 1)

  @doc """
  Create matrix of random floats in [0, 1] range.
  """
  @spec random(non_neg_integer, non_neg_integer) :: Matrex.t()
  def random(rows, columns) when is_integer(rows) and is_integer(columns),
    do: %Matrex{data: NIFs.random(rows, columns)}

  @doc """
  Create square matrix of random floats.
  """
  @spec random(non_neg_integer) :: Matrex.t()
  def random(size) when is_integer(size), do: random(size, size)

  @doc """
  Return matrix row as list by zero-based index.
  """
  @spec row_to_list(Matrex.t(), non_neg_integer) :: list(float)
  def row_to_list(%Matrex{data: matrix}, row) when is_integer(row),
    do: NIFs.row_to_list(matrix, row)

  @doc """
  Get row of matrix as matrix (vector) in binary form.
  """
  @spec row(Matrex.t(), non_neg_integer) :: Matrex.t()
  def row(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        row
      )
      when is_integer(row) and row < rows,
      do: %Matrex{
        data:
          <<1::unsigned-integer-little-32, columns::unsigned-integer-little-32,
            binary_part(data, row * columns * 4, columns * 4)::binary>>
      }

  @doc """
  Get row of matrix as list of floats
  """
  @spec row_as_list(Matrex.t(), non_neg_integer) :: list(float)
  def row_as_list(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            data::binary
          >>
        },
        row
      )
      when is_integer(row) and row < rows,
      do: binary_part(data, row * columns * 4, columns * 4) |> to_list_of_floats()

  @doc """
  Return size of matrix as {rows, cols}
  """
  @spec size(Matrex.t()) :: {non_neg_integer, non_neg_integer}
  def size(%Matrex{
        data: <<
          rows::unsigned-integer-little-32,
          cols::unsigned-integer-little-32,
          _rest::binary
        >>
      }),
      do: {rows, cols}

  @doc """
  Substracts two matrices
  """
  @spec substract(Matrex.t(), Matrex.t()) :: Matrex.t()
  def substract(%Matrex{data: first}, %Matrex{data: second}),
    do: %Matrex{data: NIFs.substract(first, second)}

  @doc """
  Substracts the second matrix from the first
  """
  @spec substract_inverse(Matrex.t(), Matrex.t()) :: Matrex.t()
  def substract_inverse(%Matrex{} = first, %Matrex{} = second), do: substract(second, first)

  @doc """
  Sums all elements.
  """
  @spec sum(Matrex.t()) :: float
  def sum(%Matrex{data: matrix}), do: NIFs.sum(matrix)

  @doc """
  Converts to flat list
  """
  @spec to_list(Matrex.t()) :: list(float)
  def to_list(%Matrex{data: matrix}), do: NIFs.to_list(matrix)

  def to_list2(<<_rows::integer-little-32, _cols::integer-little-32, data::binary>>),
    do: to_list_of_floats(data)

  defp to_list_of_floats(<<elem::float-little-32, rest::binary>>),
    do: [elem | to_list_of_floats(rest)]

  defp to_list_of_floats(<<>>), do: []

  @doc """
  Converts to list of lists
  """
  @spec to_list_of_lists(Matrex.t()) :: list(list(float))
  def to_list_of_lists(%Matrex{data: matrix}), do: NIFs.to_list_of_lists(matrix)

  @doc """
  Transposes a matrix
  """
  @spec transpose(Matrex.t()) :: Matrex.t()
  def transpose(%Matrex{data: matrix}), do: %Matrex{data: NIFs.transpose(matrix)}

  @doc """
  Create matrix of zeros of the specified size.
  """
  @spec zeros(non_neg_integer, non_neg_integer) :: Matrex.t()
  def zeros(rows, cols) when is_integer(rows) and is_integer(cols),
    do: %Matrex{data: NIFs.zeros(rows, cols)}

  @doc """
  Create square matrix of zeros
  """
  @spec zeros(non_neg_integer) :: Matrex.t()
  def zeros(size), do: zeros(size, size)
end
