defmodule Matrex.IDX do
  @moduledoc false

  # IDX format data types
  @unsigned_byte 0x08
  @signed_byte 0x09
  @short 0x0B
  @integer 0x0C
  @float 0x0D
  @double 0x0E

  defp from_idx_type(@unsigned_byte), do: :byte
  defp from_idx_type(@signed_byte), do: :byte
  defp from_idx_type(@short), do: :int16
  defp from_idx_type(@integer), do: :int32
  defp from_idx_type(@float), do: :float32
  defp from_idx_type(@double), do: :float64
  defp from_idx_type(idx_type), do: "Unsupported IDX data type: #{idx_type}"

  defp to_idx_type(:byte), do: @unsigned_byte
  defp to_idx_type(:byte), do: @signed_byte
  defp to_idx_type(:int16), do: @short
  defp to_idx_type(:int32), do: @integer
  defp to_idx_type(:float32), do: @float
  defp to_idx_type(:float64), do: @double
  defp to_idx_type(type), do: "Unsupported data type: #{type}"

  @spec load(binary) :: binary
  def load(data) when is_binary(data) do
    <<0, 0, data_type, dimensions_count>> = binary_part(data, 0, 4)

    shape =
      data
      |> binary_part(4, dimensions_count * 4)
      |> binary_to_list_of_integers()
      |> List.to_tuple()

    idx_data =
      binary_part(data, 4 + dimensions_count * 4, byte_size(data) - (4 + dimensions_count * 4))

    %Matrex{
      data: idx_data,
      shape: shape,
      strides: Matrex.strides(shape, from_idx_type(data_type)),
      type: from_idx_type(data_type)
    }
  end

  def read!(file_name) do
    {:ok, file} = File.open(file_name, :binary)

    <<0, 0, data_type, dimensions>> = IO.binread(file, 4)

    shape =
      file
      |> IO.binread(dimensions * 4)
      |> binary_to_list_of_integers()
      |> List.to_tuple()

    idx_data = IO.binread(file, :all)
    File.close(file)

    %Matrex{
      data: idx_data,
      shape: shape,
      strides: Matrex.strides(shape, from_idx_type(data_type)),
      type: from_idx_type(data_type)
    }
  end

  def write!(%Matrex{data: data, shape: shape, type: type}, file_name) do
    # File.write!(
    #   file_name,
    #   <<0, 0, to_idx_type(type), tuple_size(shape),
    #     list_of_integers_to_binary(Tuple.to_list(shape), <<>>)::binary, data::binary>>,
    #   [:binary, :write]
    # )

    {:ok, file} = File.open(file_name, [:binary, :write])
    IO.binwrite(file, <<0, 0, to_idx_type(type), tuple_size(shape)>>)
    IO.binwrite(file, list_of_integers_to_binary(Tuple.to_list(shape), <<>>))
    IO.binwrite(file, data)
    File.close(file)
  end

  def idx_to_float_binary(result, <<>>, _), do: result

  def idx_to_float_binary(result, <<elem::unsigned-integer-8, rest::binary>>, @unsigned_byte),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @unsigned_byte)

  def idx_to_float_binary(result, <<elem::signed-integer-8, rest::binary>>, @signed_byte),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @signed_byte)

  def idx_to_float_binary(result, <<elem::unsigned-integer-big-16, rest::binary>>, @short),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @short)

  def idx_to_float_binary(result, <<elem::integer-big-32, rest::binary>>, @integer),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @integer)

  def idx_to_float_binary(result, <<elem::float-big-32, rest::binary>>, @float),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @float)

  def idx_to_float_binary(result, <<elem::float-big-64, rest::binary>>, @double),
    do: idx_to_float_binary(<<result::binary, elem::float-little-32>>, rest, @double)

  defp binary_to_list_of_integers(binary, init \\ [])
  defp binary_to_list_of_integers(<<>>, list), do: Enum.reverse(list)

  defp binary_to_list_of_integers(<<value::unsigned-integer-big-32, rest::binary>>, list),
    do: binary_to_list_of_integers(rest, [value | list])

  defp list_of_integers_to_binary([], bin), do: bin

  defp list_of_integers_to_binary([h | t], bin),
    do: list_of_integers_to_binary(t, <<h::unsigned-integer-big-32, bin::binary>>)
end
