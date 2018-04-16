defmodule Matrex do
  @moduledoc """
  Performs fast operations on matrices using native C code and CBLAS library.
  """

  @on_load :load_nifs

  @doc false
  @spec load_nifs :: :ok
  def load_nifs do
    priv_dir =
      case :code.priv_dir(__MODULE__) do
        {:error, _} ->
          ebin_dir = :code.which(__MODULE__) |> :filename.dirname()
          app_path = :filename.dirname(ebin_dir)
          :filename.join(app_path, "priv")

        path ->
          path
      end

    :ok = :erlang.load_nif(:filename.join(priv_dir, "matrix_nifs"), 0)
  end

  @doc """
  Adds two matrices
  """
  @spec add(binary, binary) :: binary
  def add(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Applies the given function on each element of the matrix
  """
  @spec apply(binary, function) :: binary
  def apply(matrix, function)
      when is_binary(matrix) and is_function(function, 1) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      data::binary
    >> = matrix

    initial = <<rows::float-little-32, columns::float-little-32>>

    apply_on_matrix(data, function, initial)
  end

  @spec apply(binary, function) :: binary
  def apply(matrix, function)
      when is_binary(matrix) and is_function(function, 2) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      data::binary
    >> = matrix

    initial = <<rows::float-little-32, columns::float-little-32>>
    size = rows * columns

    apply_on_matrix(data, function, 1, size, initial)
  end

  @spec apply(binary, function) :: binary
  def apply(matrix, function)
      when is_binary(matrix) and is_function(function, 3) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      data::binary
    >> = matrix

    initial = <<rows::float-little-32, columns::float-little-32>>

    apply_on_matrix(data, function, 1, 1, columns, initial)
  end

  defp apply_on_matrix(<<>>, _, accumulator), do: accumulator

  defp apply_on_matrix(data, function, accumulator) do
    <<value::float-little-32, rest::binary>> = data

    new_value = function.(value)
    new_element = <<new_value::float-little-32>>

    apply_on_matrix(rest, function, accumulator <> new_element)
  end

  defp apply_on_matrix(<<>>, _, _, _, accumulator), do: accumulator

  defp apply_on_matrix(data, function, index, size, accumulator) do
    <<value::float-little-32, rest::binary>> = data

    new_value = function.(value, index)
    new_element = <<new_value::float-little-32>>
    new_accumulator = accumulator <> new_element

    apply_on_matrix(rest, function, index + 1, size, new_accumulator)
  end

  defp apply_on_matrix(<<>>, _, _, _, _, accumulator), do: accumulator

  defp apply_on_matrix(data, function, row_index, column_index, columns, accumulator) do
    <<value::float-little-32, rest::binary>> = data

    new_value = function.(value, row_index, column_index)
    new_element = <<new_value::float-little-32>>
    new_accumulator = accumulator <> new_element

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
  @spec apply(binary, binary, function) :: binary
  def apply(first, second, function) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      first_data::binary
    >> = first

    <<
      _::float-little-32,
      _::float-little-32,
      second_data::binary
    >> = second

    initial = <<rows::float-little-32, columns::float-little-32>>

    apply_on_matrices(first_data, second_data, function, initial)
  end

  defp apply_on_matrices(<<>>, <<>>, _, accumulator), do: accumulator

  defp apply_on_matrices(first_data, second_data, function, accumulator) do
    <<first_value::float-little-32, first_rest::binary>> = first_data
    <<second_value::float-little-32, second_rest::binary>> = second_data

    new_value = function.(first_value, second_value)
    new_element = <<new_value::float-little-32>>
    new_accumulator = accumulator <> new_element

    apply_on_matrices(first_rest, second_rest, function, new_accumulator)
  end

  @doc """
  Returns the index of the biggest element.
  """
  @spec argmax(binary) :: non_neg_integer
  def argmax(_matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    :rand.uniform()
  end

  @doc """
  Get element of a matrix at given zero-based position.
  """
  @spec at(binary, integer, integer) :: float
  def at(matrix, row, col) when is_binary(matrix) and is_integer(row) and is_integer(col) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      data::binary
    >> = matrix

    if row < 0 or row >= rows,
      do: raise(ArgumentError, message: "Row position out of range: #{row}")

    if col < 0 or col >= columns,
      do: raise(ArgumentError, message: "Column position out of range: #{col}")

    <<elem::float-little-32>> = binary_part(data, trunc(row * columns + col) * 4, 4)
    elem
  end

  @doc """
  Divides two matrices
  """
  @spec divide(binary, binary) :: binary
  def divide(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Matrix multiplication
  """
  @spec dot(binary, binary) :: binary
  def dot(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Matrix multiplication
  """
  @spec dot_and_add(binary, binary, binary) :: binary
  def dot_and_add(first, second, third)
      when is_binary(first) and is_binary(second) and is_binary(third) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Matrix multiplication where the second matrix needs to be transposed.
  """
  @spec dot_nt(binary, binary) :: binary
  def dot_nt(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Matrix multiplication where the first matrix needs to be transposed.
  """
  @spec dot_tn(binary, binary) :: binary
  def dot_tn(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create eye square matrix of given size
  """
  @spec eye(integer) :: binary
  def eye(size)
      when is_integer(size) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create matrix filled with given value
  """
  @spec fill(integer, integer, integer) :: binary
  def fill(rows, cols, value)
      when is_integer(rows) and is_integer(cols) and is_integer(value) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create square matrix filled with given value
  """
  @spec fill(integer, integer) :: binary
  def fill(size, value), do: fill(size, size, value)

  @doc """
  Return first element of a matrix.
  """
  @spec first(binary) :: float
  def first(matrix) do
    <<
      _rows::float-little-32,
      _columns::float-little-32,
      element::float-little-32,
      _rest::binary
    >> = matrix

    element
  end

  @doc """
  Displays a visualization of the matrix.
  """
  @spec inspect(binary) :: binary
  def inspect(matrix) do
    <<
      rows::float-little-32,
      columns::float-little-32,
      rest::binary
    >> = matrix

    IO.puts("Rows: #{trunc(rows)} Columns: #{trunc(columns)}")

    inspect_element(1, columns, rest)

    matrix
  end

  defp inspect_element(_, _, <<>>), do: :ok

  defp inspect_element(column, columns, elements) do
    <<element::float-little-32, rest::binary>> = elements

    next_column =
      case column == columns do
        true ->
          IO.puts(undot(element))

          1.0

        false ->
          IO.write("#{undot(element)} ")

          column + 1.0
      end

    inspect_element(next_column, columns, rest)
  end

  defp undot(f) when f - trunc(f) == 0.0, do: trunc(f)
  defp undot(f), do: f

  @doc """
  Creates "magic" n*n matrix, where sums of all dimensions are equal
  """
  @spec magic(integer) :: binary
  def magic(n) when is_integer(n) and n >= 3 do
    Matrex.MagicSquare.new(n) |> new()
  end

  @doc """
  Maximum element in a matrix.
  """
  @spec max(binary) :: number
  def max(_matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    :rand.uniform()
  end

  @doc """
  Elementwise multiplication of two matrices
  """
  @spec multiply(binary, binary) :: binary
  def multiply(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Elementwise multiplication of a scalar
  """
  @spec multiply_with_scalar(binary, number) :: binary
  def multiply_with_scalar(matrix, scalar)
      when is_binary(matrix) and is_number(scalar) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Creates a new matrix with values provided by the given function
  """
  @spec new(non_neg_integer, non_neg_integer, function) :: binary
  def new(rows, columns, function) when is_function(function, 0) do
    initial = <<rows::float-little-32, columns::float-little-32>>

    new_matrix_from_function(rows * columns, function, initial)
  end

  @spec new(non_neg_integer, non_neg_integer, function) :: binary
  def new(rows, columns, function) when is_function(function, 2) do
    initial = <<rows::float-little-32, columns::float-little-32>>
    size = rows * columns

    new_matrix_from_function(size, rows, columns, function, initial)
  end

  @spec new(non_neg_integer, non_neg_integer, list(list)) :: binary
  def new(rows, columns, list_of_lists) when is_list(list_of_lists) do
    initial = <<rows::float-little-32, columns::float-little-32>>

    Enum.reduce(list_of_lists, initial, fn list, accumulator ->
      accumulator <>
        Enum.reduce(list, <<>>, fn element, partial ->
          partial <> <<element::float-little-32>>
        end)
    end)
  end

  @doc """
  Creates new matrix from list of lists.
  """
  @spec new(list(list)) :: binary
  def new(list_of_lists) when is_list(list_of_lists) do
    rows = length(list_of_lists)
    cols = length(List.first(list_of_lists))
    new(rows, cols, list_of_lists)
  end

  defp new_matrix_from_function(0, _, accumulator), do: accumulator

  defp new_matrix_from_function(size, function, accumulator) do
    current = <<function.()::float-little-32>>

    new_matrix_from_function(size - 1, function, accumulator <> current)
  end

  defp new_matrix_from_function(0, _, _, _, accumulator), do: accumulator

  defp new_matrix_from_function(size, rows, columns, function, accumulator) do
    {row, col} =
      if rem(size, columns) == 0 do
        {rows - div(size, columns), 0}
      else
        {rows - 1 - div(size, columns), columns - rem(size, columns)}
      end

    current = <<function.(row, col)::float-little-32>>
    new_accumulator = accumulator <> current

    new_matrix_from_function(size - 1, rows, columns, function, new_accumulator)
  end

  @doc """
  Create matrix filled with ones.
  """
  def ones(rows, cols), do: fill(rows, cols, 1)

  @doc """
  Create square matrix filled with ones.
  """
  def ones(size), do: fill(size, 1)

  @doc """
  Create matrix of random floats in [0, 1] range.
  """
  @spec random(integer, integer) :: binary
  def random(rows, cols)
      when is_integer(rows) and is_integer(cols) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create square matrix of random floats.
  """
  def random(size), do: random(size, size)

  @doc """
  Return size of matrix as {rows, cols}
  """
  @spec size(binary) :: {integer, integer}
  def size(matrix) when is_binary(matrix) do
    <<
      rows::float-little-32,
      cols::float-little-32,
      _rest::binary
    >> = matrix

    {trunc(rows), trunc(cols)}
  end

  @doc """
  Substracts two matrices
  """
  @spec substract(binary, binary) :: binary
  def substract(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Substracts the second matrix from the first
  """
  @spec substract_inverse(binary, binary) :: binary
  def substract_inverse(first, second) do
    substract(second, first)
  end

  @doc """
  Sums all elements.
  """
  @spec sum(binary) :: number
  def sum(_matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    :rand.uniform()
  end

  @doc """
  Converts to flat list
  """
  @spec to_list(binary) :: binary
  def to_list(matrix) when is_binary(matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Converts to list of lists
  """
  @spec to_list_of_lists(binary) :: binary
  def to_list_of_lists(matrix) when is_binary(matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Transposes a matrix
  """
  @spec transpose(binary) :: binary
  def transpose(matrix) when is_binary(matrix) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create matrix of zeros of the specified size.
  """
  @spec zeros(integer, integer) :: binary
  def zeros(rows, cols) when is_integer(rows) and is_integer(cols) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  @doc """
  Create square matrix of zeros
  """
  @spec zeros(integer) :: binary
  def zeros(size), do: zeros(size, size)
end
