defmodule Matrex.Inspect do
  @moduledoc false

  def inspect(%Matrex{} = matrex) do
    IO.puts(do_inspect(matrex))
    matrex
  end

  def do_inspect(matrex, screen_width \\ 80)

  def do_inspect(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            _rest::binary
          >>
        } = matrex,
        screen_width
      )
      when columns < screen_width / 8 and rows <= 21 do
    rows_as_strings =
      for(
        row <- 1..rows,
        do:
          row_to_list_of_binaries(matrex, row)
          |> Enum.map(&format_float(&1))
          |> Enum.join()
      )

    top_row_length = String.length(List.first(rows_as_strings) || "") + 1
    bottom_row_length = String.length(List.last(rows_as_strings) || "") + 1
    top = "#{IO.ANSI.white()}┌#{String.pad_trailing("", top_row_length)}┐\n│#{IO.ANSI.yellow()}"
    bottom = " #{IO.ANSI.white()}│\n└#{String.pad_trailing("", bottom_row_length)}┘"
    contents_str = rows_as_strings |> Enum.join(IO.ANSI.white() <> " │\n│" <> IO.ANSI.yellow())
    "#{header(rows, columns)}\n#{top}#{contents_str}#{bottom}"
  end

  @element_byte_size 4
  @element_chars_size 8

  def do_inspect(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            body::binary
          >>
        } = matrex,
        screen_width
      )
      when columns >= screen_width / 8 or rows > 21 do
    suffix_size = prefix_size = div(screen_width, 16)

    rows_as_strings =
      for row <- displayable_rows(rows),
          do: format_row(matrex, row, rows, columns, suffix_size, prefix_size)

    rows_as_strings =
      case suffix_size + prefix_size < columns do
        true ->
          [
            binary_part(body, 0, prefix_size * @element_byte_size)
            |> format_row_head_tail(0, prefix_size)
            | rows_as_strings
          ]

        _ ->
          rows_as_strings
      end

    row_length = row_length(columns, suffix_size, prefix_size)
    half_row_length = div(row_length, 2)
    top = "#{IO.ANSI.white()}┌#{String.pad_trailing("", row_length)}┐\n│#{IO.ANSI.yellow()}"
    bottom = " #{IO.ANSI.white()}│\n└#{String.pad_trailing("", row_length)}┘"

    contents_str =
      rows_as_strings
      |> Enum.join(joiner(columns, suffix_size, prefix_size))
      |> insert_vertical_ellipsis_row(half_row_length, columns, suffix_size, prefix_size)

    "#{header(rows, columns)}\n#{top}#{contents_str}#{bottom}"
  end

  defp displayable_rows(rows) when rows > 21,
    do: Enum.to_list(1..10) ++ [-1] ++ Enum.to_list((rows - 9)..rows)

  defp displayable_rows(rows), do: 1..rows

  # Put the vertical ellipsis marker, which we will use later to insert full ellipsis row
  defp format_row(_matrix, -1, _rows, _columns, _, _), do: "⋮"

  defp format_row(%Matrex{data: matrix}, row, rows, columns, suffix_size, prefix_size)
       when row == rows and suffix_size + prefix_size < columns do
    n = chunk_offset(row, columns, suffix_size)

    binary_part(matrix, n, suffix_size * @element_byte_size)
    |> format_row_head_tail(suffix_size, 0)
  end

  defp format_row(%Matrex{} = matrex, row, _rows, columns, suffix_size, prefix_size)
       when suffix_size + prefix_size >= columns do
    matrex
    |> row_to_list_of_binaries(row)
    |> Enum.map(&format_float(&1))
    |> Enum.join()
  end

  defp row_to_list_of_binaries(
         %Matrex{
           data: <<
             rows::unsigned-integer-little-32,
             columns::unsigned-integer-little-32,
             data::binary
           >>
         },
         row
       )
       when row <= rows do
    0..(columns - 1)
    |> Enum.map(fn c ->
      binary_part(data, ((row - 1) * columns + c) * @element_byte_size, @element_byte_size)
    end)
  end

  defp format_row(%Matrex{data: matrix}, row, _rows, columns, suffix_size, prefix_size)
       when is_binary(matrix) do
    n = chunk_offset(row, columns, suffix_size)

    binary_part(matrix, n, (suffix_size + prefix_size) * @element_byte_size)
    |> format_row_head_tail(suffix_size, prefix_size)
  end

  defp chunk_offset(row, columns, suffix_size),
    do: (2 + ((row - 1) * columns + (columns - suffix_size))) * @element_byte_size

  defp format_row_head_tail(<<>>, _, _), do: <<>>

  defp format_row_head_tail(<<val::binary-4, rest::binary>>, 1, prefix_size)
       when prefix_size > 0 do
    <<format_float(val) <> IO.ANSI.white() <> " │\n│" <> IO.ANSI.yellow(),
      format_row_head_tail(rest, 0, prefix_size)::binary>>
  end

  defp format_row_head_tail(<<val::binary-4, rest::binary>>, 0, prefix_size)
       when prefix_size > 0 do
    <<
      format_float(val)::binary,
      # Separate for investigation
      format_row_head_tail(rest, 0, prefix_size - 1)::binary
    >>
  end

  defp format_row_head_tail(<<val::binary-4, rest::binary>>, suffix_size, prefix_size)
       when suffix_size > 0 do
    <<format_float(val)::binary,
      format_row_head_tail(rest, suffix_size - 1, prefix_size)::binary>>
  end

  @not_a_number <<0, 0, 192, 255>>
  @positive_infinity <<0, 0, 128, 127>>
  @negative_infinity <<0, 0, 128, 255>>

  defp format_float(@not_a_number), do: String.pad_leading("NaN", @element_chars_size)
  defp format_float(@positive_infinity), do: String.pad_leading("∞", @element_chars_size)
  defp format_float(@negative_infinity), do: String.pad_leading("-∞", @element_chars_size)
  defp format_float(<<f::float-little-32>>), do: format_float(f)

  defp format_float(f) when is_float(f) do
    f
    |> Float.round(5)
    |> Float.to_string()
    |> String.pad_leading(@element_chars_size)
  end

  defp insert_vertical_ellipsis_row(
         matrix_as_string,
         _half_row_length,
         columns,
         suffix_size,
         prefix_size
       )
       when suffix_size + prefix_size >= columns do
    String.replace(
      matrix_as_string,
      ~r/\n.*⋮.*\n/,
      "\n│#{String.pad_leading("", row_length(columns, suffix_size, prefix_size), "     ⋮  ")}│\n"
    )
  end

  defp insert_vertical_ellipsis_row(
         matrix_as_string,
         half_row_length,
         _columns,
         _suffix_size,
         _prefix_size
       ) do
    String.replace(
      matrix_as_string,
      ~r/\n.*⋮.*\n/,
      "\n│#{String.pad_leading("", half_row_length, "     ⋮  ")}… #{
        String.pad_trailing("", half_row_length - 1, "     ⋮  ")
      }│\n"
    )
  end

  defp row_length(columns, suffix_size, prefix_size) when suffix_size + prefix_size >= columns,
    do: @element_chars_size * columns + 1

  defp row_length(_columns, suffix_size, prefix_size),
    do: @element_chars_size * (suffix_size + prefix_size) + 5

  defp joiner(columns, suffix_size, prefix_size) when suffix_size + prefix_size >= columns,
    do: " #{IO.ANSI.white()}│\n│#{IO.ANSI.yellow()}"

  defp joiner(_, _, _), do: IO.ANSI.white() <> "  … " <> IO.ANSI.yellow()

  defp header(rows, columns),
    do:
      "#{IO.ANSI.white()}#Matrex#{IO.ANSI.light_white()}[#{IO.ANSI.yellow()}#{rows}#{
        IO.ANSI.light_white()
      }×#{IO.ANSI.yellow()}#{columns}#{IO.ANSI.light_white()}]"
end
