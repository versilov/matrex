defimpl Inspect, for: Matrex do
  def inspect(
        %Matrex{
          data:
            <<
              rows::unsigned-integer-little-32,
              columns::unsigned-integer-little-32,
              _rest::binary
            >> = data
        },
        %{width: screen_width} = _opts
      )
      when columns < screen_width / 8 and rows <= 50 do
    rows_as_strings =
      for(
        row <- 1..rows,
        do:
          Matrex.row_as_list(data, row - 1)
          |> Enum.map(&format_float(&1))
          |> Enum.join()
      )

    top_row_length = String.length(List.first(rows_as_strings) || "") + 1
    bottom_row_length = String.length(List.last(rows_as_strings) || "") + 1
    top = "#{IO.ANSI.white()}┌#{String.pad_trailing("", top_row_length)}┐\n│#{IO.ANSI.yellow()}"
    bottom = " #{IO.ANSI.white()}│\n└#{String.pad_trailing("", bottom_row_length)}┘"
    contents_str = rows_as_strings |> Enum.join(IO.ANSI.white() <> " │\n│" <> IO.ANSI.yellow())
    "#Matrix[#{rows}x#{columns}]\n#{top}#{contents_str}#{bottom}"
  end

  @element_byte_size 4

  def inspect(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            body::binary
          >>
        },
        %{width: screen_width} = _opts
      )
      when columns >= screen_width / 8 or rows > 50 do
    suffix_size = prefix_size = div(screen_width, 16)
    element_chars_size = 8

    rows_as_strings =
      for row <- displayable_rows(rows),
          do: format_row(body, row, rows, columns, suffix_size, prefix_size)

    rows_as_strings = [
      binary_part(body, 0, prefix_size * @element_byte_size)
      |> format_row_head_tail(0, prefix_size)
      | rows_as_strings
    ]

    row_length = element_chars_size * (suffix_size + prefix_size) + 5
    half_row_length = div(row_length, 2)
    top = "#{IO.ANSI.white()}┌#{String.pad_trailing("", row_length)}┐\n│#{IO.ANSI.yellow()}"
    bottom = " #{IO.ANSI.white()}│\n└#{String.pad_trailing("", row_length)}┘"

    contents_str =
      rows_as_strings
      |> Enum.join(IO.ANSI.white() <> "  … " <> IO.ANSI.yellow())
      |> String.replace(
        ~r/\n.*⋮.*\n/,
        "\n│#{String.pad_leading("", half_row_length, "     ⋮  ")}… #{
          String.pad_trailing("", half_row_length - 1, "     ⋮  ")
        }│\n"
      )

    "#{IO.ANSI.white()}#Matrix#{IO.ANSI.light_white()}[#{IO.ANSI.yellow()}#{rows}#{
      IO.ANSI.light_white()
    }×#{IO.ANSI.yellow()}#{columns}#{IO.ANSI.light_white()}]\n#{top}#{contents_str}#{bottom}"
  end

  defp displayable_rows(rows) when rows > 50,
    do: Enum.to_list(0..25) ++ [-1] ++ Enum.to_list((rows - 25)..(rows - 1))

  defp displayable_rows(rows), do: 0..(rows - 1)

  defp format_row(_body, -1, _rows, _columns, _, _), do: "⋮"

  defp format_row(body, row, rows, columns, suffix_size, prefix_size) when row == rows - 1 do
    n = chunk_offset(row, columns, suffix_size)

    binary_part(body, n, suffix_size * @element_byte_size)
    |> format_row_head_tail(suffix_size, 0)
  end

  defp format_row(body, row, rows, columns, suffix_size, prefix_size) do
    n = chunk_offset(row, columns, suffix_size)

    binary_part(body, n, (suffix_size + prefix_size) * @element_byte_size)
    |> format_row_head_tail(suffix_size, prefix_size)
  end

  defp chunk_offset(row, columns, suffix_size),
    do: (row * columns + (columns - suffix_size)) * @element_byte_size

  defp format_row_head_tail(<<>>, _, _), do: <<>>

  defp format_row_head_tail(<<val::float-little-32, rest::binary>>, 1, prefix_size)
       when prefix_size > 0 do
    <<format_float(val) <> IO.ANSI.white() <> " │\n│" <> IO.ANSI.yellow(),
      format_row_head_tail(rest, 0, prefix_size)::binary>>
  end

  defp format_row_head_tail(<<val::float-little-32, rest::binary>>, 0, prefix_size)
       when prefix_size > 0 do
    <<
      format_float(val)::binary,
      # Separate for investigation
      format_row_head_tail(rest, 0, prefix_size - 1)::binary
    >>
  end

  defp format_row_head_tail(<<val::float-little-32, rest::binary>>, suffix_size, prefix_size)
       when suffix_size > 0 do
    <<format_float(val)::binary,
      format_row_head_tail(rest, suffix_size - 1, prefix_size)::binary>>
  end

  defp format_float(f) when is_float(f) do
    f
    |> Float.round(5)
    |> Float.to_string()
    |> String.pad_leading(8)
  end
end
