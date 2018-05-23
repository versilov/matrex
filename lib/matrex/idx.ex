defmodule Matrex.IDX do
  @moduledoc false

  # IDX format data types
  @unsigned_byte 0x08
  @signed_byte 0x09
  @short 0x0B
  @integer 0x0C
  @float 0x0D
  @double 0x0E

  @spec load(binary) :: binary
  def load(data) when is_binary(data) do
    <<0, 0, data_type, dimensions_count>> = binary_part(data, 0, 4)
    dimensions = binary_part(data, 4, dimensions_count * 4) |> binary_to_list_of_integers()
    [rows | other] = dimensions
    cols = Enum.reduce(other, 1, &(&1 * &2))

    initial = <<rows::unsigned-integer-little-32, cols::unsigned-integer-little-32>>

    idx_data =
      binary_part(data, 4 + dimensions_count * 4, byte_size(data) - (4 + dimensions_count * 4))

    idx_to_float_binary(initial, idx_data, data_type)
  end

  def read!(file_name) do
    {:ok, file} = File.open(file_name)

    <<0, 0, data_type, dimensions>> = IO.binread(file, 4)
    dimensions = IO.binread(file, dimensions * 4) |> binary_to_list_of_integers()
    [rows | other] = dimensions
    cols = Enum.reduce(other, 1, &(&1 * &2))

    initial = <<rows::unsigned-integer-little-32, cols::unsigned-integer-little-32>>

    idx_data = IO.binread(file, :all)
    File.close(file)

    idx_to_float_binary(initial, idx_data, data_type)
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
end
