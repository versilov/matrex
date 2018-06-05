defmodule Matrex.Array do
  @moduledoc """
  NumPy style multidimensional array
  """
  import Matrex.Array.Macro
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

  @spec at(array, index) :: element
  def at(%Array{} = array, index) when not is_tuple(index), do: at(array, {index})

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

  defmacrop type_and_size() do
    {type, size} = Module.get_attribute(__CALLER__.module, :type_and_size)

    quote do
      size(unquote(size)) - unquote(type)() - little
    end
  end

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

    @spec at(array, tuple) :: element
    def at(%Array{data: data, strides: strides, type: @guard}, pos) when is_tuple(pos) do
      <<f::type_and_size()>> = binary_part(data, offset(strides, pos), bytesize(@guard))
      f
    end
  end

  defp list_to_binary([false | tail], :bool),
    do: <<0::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp list_to_binary([0 | tail], :bool),
    do: <<0::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp list_to_binary([_ | tail], :bool),
    do: <<1::size(1), list_to_binary(tail, :bool)::bitstring>>

  defp binary_to_text(<<e::size(1), rest::bitstring>>, :bool),
    do: "#{e} " <> binary_to_text(rest, :bool)

  def at(%Array{data: data, strides: strides, type: :bool}, pos) when is_tuple(pos) do
    off = offset(strides, pos)
    <<_::size(off), x::size(1), _rest::bitstring>> = data
    x == 1
  end

  defp random_binary(count, :bool) when count < 64, do: <<random_cell(:int64)::size(count)>>

  defp random_binary(count, :bool) do
    bits = rem(count, 64)

    Enum.reduce(1..div(count, 64), <<random_cell(:int64)::size(bits)>>, fn _, bin ->
      <<random_cell(:int64)::integer-64, bin::bitstring>>
    end)
  end

  # [
  #   float64: {:float, 64},
  #   float32: {:float, 32},
  #   byte: {:integer, 8},
  #   int16: {:integer, 16},
  #   int32: {:integer, 32},
  #   int64: {:integer, 64}
  # ]
  # |> Enum.each(fn {type, {spec, bits}} ->
  #   defp list_to_binary([e | tail], unquote(type)),
  #     do: <<e::unquote(spec)()-little-unquote(bits), list_to_binary(tail, unquote(type))::binary>>
  # end)

  # defp list_to_binary([e | tail], :float32 = type),
  #   do: <<e::float-little-32, list_to_binary(tail, type)::binary>>

  # defp list_to_binary([e | tail], :byte = type), do: <<e, list_to_binary(tail, type)::binary>>

  @spec random(tuple, type) :: array
  def random(shape, type \\ :float32) when is_tuple(shape) do
    %Array{
      data: random_binary(elements_count(shape), type),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

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

  def zeros(shape, type) when is_tuple(shape) and type in @element_types do
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
