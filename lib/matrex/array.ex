defmodule Matrex.Array do
  @moduledoc """
  NumPy style multidimensional array
  """
  alias Matrex.Array
  alias Matrex.Array.NIFs

  @enforce_keys [:data, :type, :shape, :strides]
  defstruct data: nil, type: :float32, strides: {}, shape: {}
  @element_types [:float32, :float64, :int16, :int32, :int64, :byte, :bool]

  @type element :: number | :nan | :inf | :neg_inf
  @type type :: :float32 | :float64 | :int16 | :int32 | :int64 | :byte | :bool
  @type index :: pos_integer
  @type array :: %Array{data: binary, type: atom, shape: tuple, strides: tuple}
  @type t :: array

  @spec add(array, array) :: array
  def add(%Array{data: data1, shape: shape, strides: strides, type: type}, %Array{
        data: data2,
        shape: shape,
        strides: strides,
        type: type
      }) do
    %Array{
      data: NIFs.add_arrays(data1, data2, type),
      shape: shape,
      strides: strides,
      type: type
    }
  end

  defp add_data(<<>>, <<>>, _), do: <<>>

  defp add_data(
         <<e1::float-little-32, rest1::binary>>,
         <<e2::float-little-32, rest2::binary>>,
         :float32
       ) do
    <<e1 + e2::float-little-32, add_data(rest1, rest2, :float32)::binary>>
  end

  defp add_data(
         <<e1, rest1::binary>>,
         <<e2, rest2::binary>>,
         :byte = type
       ) do
    <<e1 + e2, add_data(rest1, rest2, type)::binary>>
  end

  @spec at(array, index, index) :: element
  def at(%Array{} = array, row, col), do: at(array, {row, col})

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :float32}, pos) when is_tuple(pos) do
    <<f::float-little-32>> = binary_part(data, offset(strides, pos), bytesize(:float32))
    f
  end

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :float64}, pos) when is_tuple(pos) do
    <<f::float>> = binary_part(data, offset(strides, pos), bytesize(:float64))
    f
  end

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :byte}, pos) when is_tuple(pos) do
    <<f::size(8)>> = binary_part(data, offset(strides, pos), bytesize(:byte))
    f
  end

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
  defp binary_to_text(<<e, rest::binary>>, :byte), do: "#{e} " <> binary_to_text(rest, :byte)

  defp binary_to_text(<<e::float-little-32, rest::binary>>, :float32),
    do: "#{e} " <> binary_to_text(rest, :float32)

  defp binary_to_text(<<e::float, rest::binary>>, :float64),
    do: "#{e} " <> binary_to_text(rest, :float64)

  @spec new([element], tuple) :: array
  def new(list, shape, type \\ :float32) do
    %Array{
      data: list_to_binary(list, type),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

  defp list_to_binary([], _), do: <<>>

  [
    float64: {:float, 64},
    float32: {:float, 32},
    byte: {:integer, 8},
    int16: {:integer, 16},
    int32: {:integer, 32},
    int64: {:integer, 64}
  ]
  |> Enum.each(fn {type, {spec, bits}} ->
    defp list_to_binary([e | tail], unquote(type)),
      do: <<e::unquote(spec)()-little-unquote(bits), list_to_binary(tail, unquote(type))::binary>>
  end)

  # defp list_to_binary([e | tail], :float32 = type),
  #   do: <<e::float-little-32, list_to_binary(tail, type)::binary>>
  #
  # defp list_to_binary([e | tail], :byte = type), do: <<e, list_to_binary(tail, type)::binary>>

  @spec random(tuple, type) :: array
  def random(shape, type \\ :float32) do
    %Array{
      data: random_binary(elements_count(shape), type),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

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

  def zeros(shape, type) when is_tuple(shape) and type in @element_types do
    bitsize = elements_count(shape) * bitsize(type)

    %Array{
      data: <<0::size(bitsize)>>,
      type: type,
      shape: shape,
      strides: strides(shape, type)
    }
  end

  @spec random_binary(pos_integer, type) :: binary
  defp random_binary(count, :float32),
    do:
      Enum.reduce(1..count, <<>>, fn _, bin ->
        <<:rand.uniform()::float-little-32, bin::binary>>
      end)

  @spec random_binary(pos_integer, type) :: binary
  defp random_binary(count, :float64),
    do:
      Enum.reduce(1..count, <<>>, fn _, bin ->
        <<:rand.uniform()::float, bin::binary>>
      end)

  defp random_binary(count, :byte),
    do:
      Enum.reduce(1..count, <<>>, fn _, bin ->
        <<:rand.uniform(256) - 1, bin::binary>>
      end)

  defp range_to_binary(a..b, :float32),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::float-little-32, bin::binary>> end)

  defp range_to_binary(a..b, :float64),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::float, bin::binary>> end)

  defp range_to_binary(a..b, :int32),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::integer-little-32, bin::binary>> end)

  defp range_to_binary(a..b, :byte),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x, bin::binary>> end)

  @doc false
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

  defp bytesize(:float32), do: 4
  defp bytesize(:float64), do: 8
  defp bytesize(:int16), do: 2
  defp bytesize(:int32), do: 4
  defp bytesize(:int64), do: 8
  defp bytesize(:byte), do: 1
  defp bytesize(:bool), do: nil

  defp bitsize(:float32), do: 32
  defp bitsize(:float64), do: 64
  defp bitsize(:int16), do: 16
  defp bitsize(:int32), do: 32
  defp bitsize(:int64), do: 64
  defp bitsize(:byte), do: 8
  defp bitsize(:bool), do: 1
end
