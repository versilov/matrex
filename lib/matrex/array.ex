defmodule Matrex.Array do
  @moduledoc """
  NumPy style multidimensional array
  """
  alias Matrex.Array
  alias Matrex.Array.NIFs

  @enforce_keys [:data, :type, :shape, :strides]
  defstruct data: nil, type: :float32, strides: {}, shape: {}
  @types [:float32, :float64, :int16, :int32, :int64, :byte, :bool]

  @type element :: number | :nan | :inf | :neg_inf
  @type type :: :float32 | :float64 | :int16 | :int32 | :int64 | :byte | :bool
  @type index :: pos_integer
  @type array :: %Array{data: binary, type: atom, shape: tuple, strides: tuple}
  @type t :: array

  @spec at(array, index, index) :: element
  def at(%Array{} = array, row, col), do: at(array, {row, col})

  @spec at(array, index) :: element
  def at(%Array{} = array, index) when not is_tuple(index), do: at(array, {index})

  @behaviour Access
  @impl Access
  def fetch(array, key)

  def fetch(%Array{shape: {rows, _cols}}, :rows), do: {:ok, rows}
  def fetch(%Array{shape: {_rows, cols}}, :cols), do: {:ok, cols}
  def fetch(%Array{shape: {_rows, cols}}, :columns), do: {:ok, cols}
  def fetch(%Array{shape: shape}, :shape), do: {:ok, shape}

  @spec fill(element, tuple, type) :: array
  def fill(value, shape, type \\ :float32) do
    %Array{
      data: fill_data(value, elements_count(shape), type),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

  defp fill_data(_, 0, _), do: <<>>
  defp fill_data(value, count, :byte), do: <<value, fill_data(value, count - 1, :byte)::binary>>

  @spec inspect(array) :: array
  def inspect(%Array{data: data, type: type} = array) do
    binary_to_text(data, type)
    |> String.trim()
    |> IO.puts()

    array
  end

  defp binary_to_text(<<>>, _type), do: ""

  @spec from_string(binary, atom) :: array
  def from_string(text, type \\ :float32) when is_binary(text) and is_atom(type) do
    lol =
      text
      |> String.split(["\n", ";"], trim: true)
      |> Enum.map(fn line ->
        line
        |> String.split(["\s", ","], trim: true)
        |> Enum.map(fn f -> Matrex.parse_float(f) end)
      end)

    rows = length(lol)
    cols = length(hd(lol))

    lol
    |> List.flatten()
    |> new({rows, cols}, type)
  end

  @spec new([element], tuple, atom) :: array
  def new(list, shape, type \\ :float32) when is_list(list) and is_tuple(shape) do
    %Array{
      data: list_to_binary(list, type),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

  defp list_to_binary([], _), do: <<>>

  defmacrop type_and_size() do
    {type, size} = Module.get_attribute(__CALLER__.module, :type_and_size)

    if size == 8 do
      quote do: size(unquote(size))
    else
      quote do: size(unquote(size)) - unquote(type)() - little
    end
  end

  def dot(array1, array2, alpha \\ 1.0)

  def dot(%Array{type: type1}, %Array{type: type2}, _alpha) when type1 != type2,
    do: raise(ArgumentError, "arrays types mismatch: #{type1} vs #{type2}")

  def ones(shape, type \\ :float32)

  def set(array, row, col, value), do: set(array, {row, col}, value)

  types = [
    float64: {:float, 64},
    float32: {:float, 32},
    byte: {:integer, 8},
    int16: {:integer, 16},
    int32: {:integer, 32},
    int64: {:integer, 64}
  ]

  for {guard, type_and_size} <- types do
    @guard guard
    @type_and_size type_and_size

    def add(%Array{data: data1, shape: shape, strides: strides, type: @guard}, %Array{
          data: data2,
          shape: shape,
          strides: strides,
          type: @guard
        }) do
      %Array{
        data: apply(NIFs, :"add_arrays_#{to_string(@guard)}", [data1, data2]),
        shape: shape,
        strides: strides,
        type: @guard
      }
    end

    def add(%Array{data: data, type: @guard} = array, scalar) when is_number(scalar),
      do: %{array | data: apply(NIFs, :"add_scalar_#{to_string(@guard)}", [data, scalar])}

    @spec at(array, tuple) :: element
    def at(%Array{data: data, strides: strides, type: @guard}, pos) when is_tuple(pos) do
      <<f::type_and_size()>> = binary_part(data, offset(strides, pos), bytesize(@guard))
      f
    end

    def dot(
          %Array{data: data1, shape: {rows, dim}, strides: {stride1, _}, type: @guard},
          %Array{
            data: data2,
            shape: {dim, cols},
            strides: {_, stride2},
            type: @guard
          },
          alpha
        ) do
      %Array{
        data:
          apply(NIFs, :"dot_arrays_#{to_string(@guard)}", [data1, data2, rows, dim, cols, alpha]),
        shape: {rows, cols},
        strides: {stride1, stride2},
        type: @guard
      }
    end

    def multiply(%Array{data: data1, shape: shape, strides: strides, type: @guard}, %Array{
          data: data2,
          shape: shape,
          strides: strides,
          type: @guard
        }) do
      %Array{
        data: apply(NIFs, :"multiply_arrays_#{to_string(@guard)}", [data1, data2]),
        shape: shape,
        strides: strides,
        type: @guard
      }
    end

    @spec ones(tuple, type) :: array
    def ones(shape, @guard) when is_tuple(shape),
      do: %Array{
        data: apply(NIFs, :"ones_array_#{to_string(@guard)}", [elements_count(shape)]),
        shape: shape,
        strides: strides(shape, @guard),
        type: @guard
      }

    @spec set(array, tuple, integer | float) :: array
    def set(%Array{data: data, strides: strides, shape: shape, type: @guard} = array, pos, value) do
      offset = offset(strides, pos)

      %{
        array
        | data:
            <<binary_part(data, 0, offset)::binary, value::type_and_size(),
              binary_part(
                data,
                offset + bytesize(@guard),
                bytesize(shape, @guard) - offset - bytesize(@guard)
              )::binary>>
      }
    end

    @spec square(array) :: array
    def square(%Array{data: data, type: @guard} = array),
      do: %{array | data: apply(NIFs, :"square_array_#{to_string(@guard)}", [data])}

    def sum(%Array{data: data, type: @guard}),
      do: apply(NIFs, :"array_sum_#{to_string(@guard)}", [data])

    def to_type(%Array{type: @guard} = array, @type), do: array

    def to_type(%Array{data: data, shape: shape, type: @guard} = array, type) when type in @types,
      do: %{
        array
        | data: apply(NIFs, :"array_#{to_string(@guard)}_to_#{to_string(type)}", [data]),
          strides: strides(shape, type),
          type: type
      }

    defp list_to_binary([e | tail], @guard) do
      <<e::type_and_size(), list_to_binary(tail, @guard)::binary>>
    end

    defp binary_to_text(<<e::type_and_size(), rest::binary>>, @guard),
      do: "#{e} " <> binary_to_text(rest, @guard)

    @spec random_binary(pos_integer, type) :: binary
    defp random_binary(count, @guard),
      do:
        Enum.reduce(1..count, <<>>, fn _, bin ->
          <<random_cell(@guard)::type_and_size(), bin::binary>>
        end)

    defp range_to_binary(a..b, @guard),
      do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::type_and_size(), bin::binary>> end)
  end

  def random(shape, type \\ :float32)

  float_types = [
    float64: {:float, 64},
    float32: {:float, 32}
  ]

  for {guard, type_and_size} <- float_types do
    @guard guard
    @type_and_size type_and_size

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

    @spec apply(array, atom) :: array
    def apply(%Array{data: data, type: @guard} = array, math_func)
        when is_atom(math_func) and math_func in @math_functions,
        do: %{
          array
          | data: apply(NIFs, :"array_apply_math_#{to_string(@guard)}", [data, math_func])
        }

    @spec random(tuple, type) :: array
    def random(shape, @guard),
      do: %Array{
        data: apply(NIFs, :"random_array_#{to_string(@guard)}", [elements_count(shape)]),
        shape: shape,
        strides: strides(shape, @guard),
        type: @guard
      }
  end

  # Bool (布尔)

  def apply(%Array{data: data, type: type} = array, fun) when is_function(fun, 1),
    do: %{array | data: apply_fun(data, fun, type)}

  def apply(%Array{data: data, type: type, shape: shape} = array, fun) when is_function(fun, 2),
    do: %{array | data: apply_fun(data, fun, shape, Tuple.duplicate(1, tuple_size(shape)), type)}

  defp apply_fun(<<x::size(1), rest::bitstring>>, fun, :bool) when is_function(fun, 1),
    do: <<fun.(x)::size(1), apply_fun(rest, fun, :bool)::bitstring>>

  defp apply_fun(<<>>, _, _), do: <<>>

  defp apply_fun(<<x::size(1), rest::bitstring>>, fun, shape, pos, :bool) do
    new_pos = next_pos(pos, shape)
    <<fun.(x, pos)::size(1), apply_fun(rest, fun, shape, new_pos, :bool)::bitstring>>
  end

  defp apply_fun(<<>>, _fun, _shape, _pos, _type), do: <<>>

  defp next_pos(pos, shape) do
    Enum.reduce_while((tuple_size(shape) - 1)..0, pos, fn i, p ->
      new_coord = elem(p, i) + 1

      if new_coord <= elem(shape, i) do
        {:halt, put_elem(p, i, new_coord)}
      else
        {:cont, put_elem(p, i, 1)}
      end
    end)
  end

  def at(%Array{data: data, strides: strides, type: :bool}, pos) when is_tuple(pos) do
    off = offset(strides, pos)
    <<_::size(off), x::size(1), _rest::bitstring>> = data
    x
  end

  def heatmap(%Array{data: data, shape: {rows, cols}, type: :bool} = array) do
    for n <- 1..div(rows, 2) do
      rows_string(array, n)
    end
    |> Enum.join("\n")
    |> IO.puts()

    array
  end

  defp rows_string(%Array{shape: {rows, cols}} = array, n) do
    Enum.reduce(1..cols, <<>>, fn c, acc ->
      acc <> pixel(at(array, n * 2 - 1, c), at(array, n * 2, c))
    end)
  end

  defp pixel(1, 1), do: "█"
  defp pixel(0, 0), do: " "
  defp pixel(1, 0), do: "▀"
  defp pixel(0, 1), do: "▄"

  defp list_to_binary([false | tail], :bool),
    do: <<0::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp list_to_binary([0 | tail], :bool),
    do: <<0::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp list_to_binary([_ | tail], :bool),
    do: <<1::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp binary_to_text(<<e::size(1), rest::bitstring>>, :bool),
    do: "#{e} " <> binary_to_text(rest, :bool)

  defp random_binary(count, :bool) when count < 64, do: <<random_cell(:int64)::size(count)>>

  defp random_binary(count, :bool) do
    bits = rem(count, 64)

    Enum.reduce(1..div(count, 64), <<random_cell(:int64)::size(bits)>>, fn _, bin ->
      <<random_cell(:int64)::integer-64, bin::bitstring>>
    end)
  end

  # @spec random(tuple, type) :: array
  # def random(shape, type \\ :float32) when is_tuple(shape) do
  #   %Array{
  #     data: random_binary(elements_count(shape), type),
  #     shape: shape,
  #     strides: strides(shape, type),
  #     type: type
  #   }
  # end

  defp random_cell(:float32), do: :rand.uniform()
  defp random_cell(:float64), do: :rand.uniform()
  defp random_cell(:byte), do: :rand.uniform(256) - 1
  defp random_cell(:int32), do: :rand.uniform(65_536) - 1
  defp random_cell(:int32), do: :rand.uniform(2_147_483_647 * 2) - 1
  defp random_cell(:int64), do: :rand.uniform(18_446_744_073_709_551_615) - 1

  @spec reshape(array, tuple) :: array
  def reshape(%Array{data: data, type: type}, shape) when is_tuple(shape),
    do: %Array{data: data, shape: shape, strides: strides(shape, type), type: type}

  @spec reshape(Range.t(), tuple, type) :: array
  def reshape(a..b, shape, type \\ :float32) do
    if abs(b - a) + 1 != elements_count(shape),
      do: raise(ArgumentError, message: "range and shape do not match.")

    %Array{
      data: range_to_binary(a..b, type),
      type: type,
      shape: shape,
      strides: strides(shape, type)
    }
  end

  @spec shape(array) :: tuple
  def shape(%Array{shape: shape}), do: shape

  @spec transpose(array) :: array
  def transpose(%Array{shape: {sh1, sh2}, strides: {s1, s2}} = array),
    do: %Array{array | shape: {sh2, sh1}, strides: {s2, s1}}

  @spec zeros(tuple, type) :: array
  def zeros(shape, type \\ :float32)
  def zeros(shape, type) when is_integer(shape), do: zeros({shape}, type)

  def zeros(shape, type) when is_tuple(shape) and type in @types do
    bitsize = elements_count(shape) * bitsize(type)

    %Array{
      data: <<0::size(bitsize)>>,
      type: type,
      shape: shape,
      strides: strides(shape, type)
    }
  end

  @doc false
  def strides({_}, type), do: {bytesize(type)}

  def strides(shape, type) when is_tuple(shape) and is_atom(type) do
    Enum.reduce((tuple_size(shape) - 1)..1, {{bytesize(type)}, 1}, fn i, {strides, acc} ->
      stride = acc * elem(shape, i)
      {Tuple.insert_at(strides, 0, stride * bytesize(type)), stride}
    end)
    |> elem(0)
  end

  defp offset(strides, pos) when tuple_size(strides) == tuple_size(pos),
    do:
      Enum.reduce(0..(tuple_size(pos) - 1), 0, fn i, off ->
        elem(strides, i) * (elem(pos, i) - 1) + off
      end)

  defp elements_count(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(&(&1 * &2))
  end

  defp bytesize(shape, type) when is_tuple(shape) and type in @types,
    do: elements_count(shape) * bytesize(type)

  defp bytesize(:float32), do: 4
  defp bytesize(:float64), do: 8
  defp bytesize(:int16), do: 2
  defp bytesize(:int32), do: 4
  defp bytesize(:int64), do: 8
  defp bytesize(:byte), do: 1
  # Actually, for bool this is bitsize
  defp bytesize(:bool), do: 1

  defp bitsize(:float32), do: 32
  defp bitsize(:float64), do: 64
  defp bitsize(:int16), do: 16
  defp bitsize(:int32), do: 32
  defp bitsize(:int64), do: 64
  defp bitsize(:byte), do: 8
  defp bitsize(:bool), do: 1
end
