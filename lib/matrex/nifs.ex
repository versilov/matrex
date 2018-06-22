defmodule Matrex.NIFs do
  @moduledoc false

  # All NIFs accept zero-based matrix subscripts, while Matrex module accepts one-based subscripts.

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

    :ok = :erlang.load_nif(:filename.join(priv_dir, "nifs"), 0)
  end

  @types [
    "float64",
    "float32",
    "byte",
    "int16",
    "int32",
    "int64"
  ]

  @float_types [
    "float64",
    "float32"
  ]

  @nifs [
    add_scalar: 3,
    add: 4,
    argmax: 1,
    column_to_list: 3,
    concat_columns: 4,
    divide_by_scalar: 2,
    divide_scalar: 2,
    divide: 3,
    dot: 6,
    dot_and_add: 7,
    dot_nt: 6,
    dot_tn: 6,
    eye: 2,
    fill: 2,
    find: 2,
    from_range: 2,
    max: 1,
    min: 1,
    multiply: 3,
    multiply_with_scalar: 2,
    neg: 1,
    random: 1,
    resize: 4,
    row_to_list: 3,
    set: 3,
    set_column: 4,
    submatrix: 6,
    subtract: 4,
    subtract_from_scalar: 3,
    sum: 1,
    to_list: 1,
    to_list_of_lists: 3,
    transpose: 3,
    byte_to: 1,
    int16_to: 1,
    int32_to: 1,
    int64_to: 1,
    float32_to: 1,
    float64_to: 1
  ]

  @float_nifs [apply_math: 2, dot_and_apply: 7, normalize: 1, max_finite: 1, min_finite: 1]

  for {name, arity} <- @nifs do
    for type <- @types do
      def unquote(:"#{name}_#{type}")(
            unquote_splicing(Enum.map(1..arity, &Macro.var(:"_arg#{&1}", nil)))
          ),
          do: :erlang.nif_error(:nif_library_not_loaded)
    end
  end

  for {name, arity} <- @float_nifs do
    for type <- @float_types do
      def unquote(:"#{name}_#{type}")(
            unquote_splicing(Enum.map(1..arity, &Macro.var(:"_arg#{&1}", nil)))
          ),
          do: :erlang.nif_error(:nif_library_not_loaded)
    end
  end

  @spec zeros(integer) :: binary
  def zeros(byte_size) when is_integer(byte_size), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec add(binary, binary, number, number) :: binary
  def add(first, second, alpha, beta)
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

  @spec concat_columns(binary, binary) :: binary
  def concat_columns(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

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
  def dot_tn(first, second, alpha)
      when is_binary(first) and is_binary(second) and is_number(alpha),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec eye(pos_integer, number) :: binary
  def eye(size, value)
      when is_integer(size) and is_number(value),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec fill(non_neg_integer, non_neg_integer, non_neg_integer) :: binary
  def fill(rows, cols, value)
      when is_integer(rows) and is_integer(cols) and is_integer(value),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec find(binary, binary) :: {pos_integer, pos_integer} | nil
  def find(
        <<
          _rows::binary-4,
          columns::binary-4,
          data::binary
        >>,
        <<value::binary-4>>
      ),
      do: do_find(data, value, 0, columns)

  defp do_find(<<>>, <<_value::binary-4>>, _, _), do: nil

  defp do_find(<<elem::binary-4, _rest::binary>>, <<value::binary-4>>, index, columns)
       when elem == value,
       do: {div(index, columns), rem(index, columns)}

  defp do_find(<<_elem::binary-4, rest::binary>>, <<value::binary-4>>, index, columns),
    do: do_find(rest, value, index + 1, columns)

  @spec from_range(pos_integer, pos_integer, pos_integer, pos_integer) :: binary
  def from_range(from, to, rows, cols)
      when is_integer(from) and is_integer(to) and is_integer(rows) and is_integer(cols),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec max(binary) :: float
  def max(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec min(binary) :: float
  def min(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec max_finite(binary) :: float
  def max_finite(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

  @spec min_finite(binary) :: float
  def min_finite(_matrix), do: :erlang.nif_error(:nif_library_not_loaded)

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

  @spec normalize(binary) :: binary
  def normalize(matrex) do
    mn = min(matrex)
    mx = max(matrex)
    range = mx - mn

    Matrex.apply(%Matrex{data: matrex, shape: {}, strides: {}, type: :float32}, fn x ->
      (x - mn) / range
    end).data
  end

  @spec random(non_neg_integer, non_neg_integer) :: binary
  def random(rows, cols)
      when is_integer(rows) and is_integer(cols),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec resize(binary, number) :: binary
  def resize(matrex, scale) when is_binary(matrex) and is_number(scale),
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

  @spec set(binary, non_neg_integer, non_neg_integer, binary) :: binary
  def set(
        <<
          rows::unsigned-integer-little-32,
          cols::unsigned-integer-little-32,
          data::binary
        >>,
        row,
        column,
        <<value::binary-4>>
      ) do
    pos = row * cols + column

    <<rows::unsigned-integer-little-32, cols::unsigned-integer-little-32,
      binary_part(data, 0, pos * 4)::binary, value::binary-4,
      binary_part(data, (pos + 1) * 4, (rows * cols - pos - 1) * 4)::binary>>
  end

  @spec set_column(binary, non_neg_integer, binary) :: binary
  def set_column(matrex, column, column_matrex)
      when is_binary(matrex) and is_integer(column) and is_binary(column_matrex),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec submatrix(binary, pos_integer, pos_integer, pos_integer, pos_integer) :: binary
  def submatrix(matrex, row_from, row_to, col_from, col_to)
      when is_binary(matrex) and is_integer(row_from) and is_integer(row_to) and
             is_integer(col_from) and is_integer(col_to),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec subtract(binary, binary) :: binary
  def subtract(first, second)
      when is_binary(first) and is_binary(second),
      do: :erlang.nif_error(:nif_library_not_loaded)

  @spec subtract_from_scalar(number, binary) :: binary
  def subtract_from_scalar(
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
end
