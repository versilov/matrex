defmodule Matrex.Array do
  @moduledoc """
  NumPy style multidimensional array
  """
  alias Matrex.Array

  @enforce_keys [:data, :type, :shape, :strides]
  defstruct data: nil, type: :float, strides: {}, shape: {}
  @element_types [:float, :double, :integer, :byte, :bool]

  @type element :: number | :nan | :inf | :neg_inf
  @type type :: :float | :double | :integer | :byte | :bool
  @type index :: pos_integer
  @type array :: %Array{data: binary, type: atom, shape: tuple, strides: tuple}
  @type t :: array

  @spec at(array, index, index) :: element
  def at(%Array{} = array, row, col), do: at(array, {row, col})

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :float}, pos) when is_tuple(pos) do
    <<f::float-little-32>> = binary_part(data, offset(strides, pos), bytesize(:float))
    f
  end

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :double}, pos) when is_tuple(pos) do
    <<f::float>> = binary_part(data, offset(strides, pos), bytesize(:double))
    f
  end

  @spec at(array, tuple) :: element
  def at(%Array{data: data, strides: strides, type: :byte}, pos) when is_tuple(pos) do
    <<f::size(8)>> = binary_part(data, offset(strides, pos), bytesize(:byte))
    f
  end

  @spec random(tuple, type) :: array
  def random(shape, type \\ :float) do
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
  def reshape(a..b, shape, type \\ :float) do
    if b - a + 1 != elements_count(shape),
      do: raise(ArgumentError, message: "range and shape do not match.")

    %Array{
      data: range_to_binary(a..b, type),
      type: type,
      shape: shape,
      strides: strides(shape, type)
    }
  end

  def shape(%Array{shape: shape}), do: shape

  @spec zeros(tuple, type) :: array
  def zeros(shape, type \\ :float)
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
  defp random_binary(count, :float),
    do:
      Enum.reduce(1..count, <<>>, fn _, bin ->
        <<:rand.uniform()::float-little-32, bin::binary>>
      end)

  defp random_binary(count, :byte),
    do:
      Enum.reduce(1..count, <<>>, fn _, bin ->
        <<:rand.uniform(256) - 1, bin::binary>>
      end)

  defp range_to_binary(a..b, :float),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::float-little-32, bin::binary>> end)

  defp range_to_binary(a..b, :double),
    do: Enum.reduce(b..a, <<>>, fn x, bin -> <<x::float, bin::binary>> end)

  defp range_to_binary(a..b, :integer),
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

  defp bytesize(:float), do: 4
  defp bytesize(:integer), do: 4
  defp bytesize(:double), do: 8
  defp bytesize(:byte), do: 1
  defp bytesize(:bool), do: nil

  defp bitsize(:float), do: 32
  defp bitsize(:integer), do: 32
  defp bitsize(:double), do: 64
  defp bitsize(:byte), do: 8
  defp bitsize(:bool), do: 1
end
