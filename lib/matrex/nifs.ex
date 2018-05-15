defmodule Matrex.NIFs do
  @moduledoc false

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

  @spec add(binary, binary) :: binary
  def add(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec add_with_scale(binary, binary, number, number) :: binary
  def add_with_scale(first, second, alpha, beta)
      when is_binary(first) and is_binary(second) and is_number(alpha) and is_number(beta),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec add_scalar(binary, number) :: binary
  def add_scalar(
        <<
          rows::unsigned-integer-little-32,
          columns::unsigned-integer-little-32,
          data::binary
        >>,
        scalar
      )
      when is_number(scalar) do
    0..(rows * columns - 1)
    |> Enum.reduce(
      <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>,
      fn index, matrix ->
        <<elem::float-little-32>> = binary_part(data, index * 4, 4)

        <<matrix::binary, elem + scalar::float-little-32>>
      end
    )
  end

  @spec apply_math(binary, atom) :: binary
  def apply_math(matrix, c_function) when is_binary(matrix) and is_atom(c_function),
    do: :erlang.nif_error(:nif_library_not_loaded)

  @spec apply_parallel_math(binary, atom) :: binary
  def apply_parallel_math(matrix, c_function) when is_binary(matrix) and is_atom(c_function),
    do: :erlang.nif_error(:nif_library_not_loaded)

  @spec argmax(binary) :: non_neg_integer
  def argmax(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec column_to_list(binary, non_neg_integer) :: [float]
  def column_to_list(
        <<
          rows::unsigned-integer-little-32,
          columns::unsigned-integer-little-32,
          data::binary
        >>,
        col
      )
      when is_integer(col) and col >= 0 and col < columns do
    0..(rows - 1)
    |> Enum.map(fn row ->
      <<elem::float-little-32>> = binary_part(data, (row * columns + col) * 4, 4)
      elem
    end)
  end

  @spec divide(binary, binary) :: binary
  def divide(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec divide_scalar(number, binary) :: binary
  def divide_scalar(scalar, matrix)
      when is_number(scalar) and is_binary(matrix),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec divide_by_scalar(binary, number) :: binary
  def divide_by_scalar(matrix, scalar)
      when is_number(scalar) and is_binary(matrix),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec dot(binary, binary) :: binary
  def dot(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec dot_and_apply(binary, binary, atom) :: binary
  def dot_and_apply(first, second, function)
      when is_binary(first) and is_binary(second) and is_atom(function),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec dot_and_add(binary, binary, binary) :: binary
  def dot_and_add(first, second, third)
      when is_binary(first) and is_binary(second) and is_binary(third),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec dot_nt(binary, binary) :: binary
  def dot_nt(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec dot_tn(binary, binary, number) :: binary
  def dot_tn(first, second, aplha), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec eye(pos_integer, number) :: binary
  def eye(size, value)
      when is_integer(size) and is_number(value),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec fill(non_neg_integer, non_neg_integer, non_neg_integer) :: binary
  def fill(rows, cols, value)
      when is_integer(rows) and is_integer(cols) and is_integer(value),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec max(binary) :: float
  def max(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec multiply(binary, binary) :: binary
  def multiply(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec multiply_with_scalar(binary, number) :: binary
  def multiply_with_scalar(matrix, scalar)
      when is_binary(matrix) and is_number(scalar),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec neg(binary) :: binary
  def neg(
        <<
          rows::unsigned-integer-little-32,
          columns::unsigned-integer-little-32,
          data::binary
        >> = matrix
      )
      when is_binary(matrix) do
    0..(rows * columns - 1)
    |> Enum.reduce(
      <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>,
      fn index, matrix ->
        <<elem::float-little-32>> = binary_part(data, index * 4, 4)

        <<matrix::binary, -elem::float-little-32>>
      end
    )
  end

  @spec random(non_neg_integer, non_neg_integer) :: binary
  def random(rows, cols)
      when is_integer(rows) and is_integer(cols),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec row_to_list(binary, non_neg_integer) :: [float]
  def row_to_list(
        <<
          rows::unsigned-integer-little-32,
          columns::unsigned-integer-little-32,
          data::binary
        >>,
        row
      )
      when is_integer(row) and row >= 0 and row < rows,
      do: binary_part(data, row * columns * 4, columns * 4) |> to_list_of_floats()

  defp to_list_of_floats(<<elem::float-little-32, rest::binary>>),
    do: [elem | to_list_of_floats(rest)]

  defp to_list_of_floats(<<>>), do: []

  @spec set(binary, non_neg_integer, non_neg_integer, number) :: binary
  def set(
        <<
          rows::unsigned-integer-little-32,
          cols::unsigned-integer-little-32,
          data::binary
        >>,
        row,
        column,
        value
      ) do
    pos = row * cols + column

    <<rows::unsigned-integer-little-32, cols::unsigned-integer-little-32,
      binary_part(data, 0, pos * 4)::binary, value::float-little-32,
      binary_part(data, (pos + 1) * 4, (rows * cols - pos - 1) * 4)::binary>>
  end

  @spec substract(binary, binary) :: binary
  def substract(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec substract_from_scalar(number, binary) :: binary
  def substract_from_scalar(
        scalar,
        <<
          rows::unsigned-integer-little-32,
          columns::unsigned-integer-little-32,
          data::binary
        >> = matrix
      )
      when is_number(scalar) and is_binary(matrix) do
    0..(rows * columns - 1)
    |> Enum.reduce(
      <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32>>,
      fn index, matrix ->
        <<elem::float-little-32>> = binary_part(data, index * 4, 4)

        <<matrix::binary, scalar - elem::float-little-32>>
      end
    )
  end

  @spec sum(binary) :: float
  def sum(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec to_list(binary) :: list(float)
  def to_list(matrix) when is_binary(matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec to_list_of_lists(binary) :: list(list(float))
  def to_list_of_lists(matrix) when is_binary(matrix),
    do: :erlang.nif_error(:nif_library_not_loaded)

  @spec transpose(binary) :: binary
  def transpose(matrix) when is_binary(matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec zeros(integer, integer) :: binary
  def zeros(rows, cols) when is_integer(rows) and is_integer(cols) do
    :erlang.nif_error(:nif_library_not_loaded)
  end
end
