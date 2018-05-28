defmodule Matrex.Inspect do
  @moduledoc false

  @ansi_formatting ~r/\e\[(\d{1,2};?)+m/

  def do_inspect(matrex, screen_width \\ 80, display_rows \\ 21)

  def do_inspect(
        %Matrex{
          data: <<
            rows::unsigned-integer-little-32,
            columns::unsigned-integer-little-32,
            _rest::binary
          >>
        } = matrex,
        screen_width,
        display_rows
      )
      when columns < (screen_width - 3) / 8 and rows <= display_rows do
    rows_as_strings =
      for(
        row <- 1..rows,
        do:
          row_to_list_of_binaries(matrex, row)
          |> Enum.map(&format_float(&1))
          |> Enum.join()
      )

    row_length =
      String.length(List.first(rows_as_strings) |> String.replace(@ansi_formatting, "") || "") + 1

    contents_str = rows_as_strings |> Enum.join(IO.ANSI.reset() <> " │\n│" <> IO.ANSI.yellow())

    contents_str = <<"│#{IO.ANSI.yellow()}", contents_str::binary, " #{IO.ANSI.reset()}│">>

    "#{header(rows, columns)}\n#{top_row(row_length)}\n#{contents_str}\n#{bottom_row(row_length)}"
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
        screen_width,
        display_rows
      )
      when columns >= (screen_width - 3) / 8 or rows > display_rows do
    available_columns = div(screen_width - 7, 8)
    prefix_size = suffix_size = div(available_columns, 2)
    prefix_size = prefix_size + rem(available_columns, 2)

    rows_as_strings =
      for row <- displayable_rows(rows, display_rows),
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
    prefix_length = prefix_size * @element_chars_size
    suffix_length = suffix_size * @element_chars_size

    contents_str =
      rows_as_strings
      |> Enum.join(joiner(columns, suffix_size, prefix_size))
      |> insert_vertical_ellipsis_row(
        prefix_length,
        suffix_length,
        columns,
        suffix_size,
        prefix_size
      )

    contents_str = <<"│#{IO.ANSI.yellow()}", contents_str::binary, " #{IO.ANSI.reset()}│">>

    "#{header(rows, columns)}\n#{top_row(row_length)}\n#{contents_str}\n#{bottom_row(row_length)}"
  end

  defp displayable_rows(rows, display_rows) when rows > display_rows,
    do:
      Enum.to_list(1..(div(display_rows, 2) + rem(display_rows, 2))) ++
        [-1] ++ Enum.to_list((rows - div(display_rows, 2))..rows)

  defp displayable_rows(rows, _), do: 1..rows

  @spec top_row(pos_integer) :: binary
  defp top_row(length), do: "#{IO.ANSI.reset()}┌#{String.pad_trailing("", length)}┐"

  @spec bottom_row(pos_integer) :: binary
  defp bottom_row(length), do: "#{IO.ANSI.reset()}└#{String.pad_trailing("", length)}┘"

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

  defp format_row(%Matrex{data: matrix}, row, _rows, columns, suffix_size, prefix_size)
       when is_binary(matrix) do
    n = chunk_offset(row, columns, suffix_size)

    binary_part(matrix, n, (suffix_size + prefix_size) * @element_byte_size)
    |> format_row_head_tail(suffix_size, prefix_size)
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

  defp chunk_offset(row, columns, suffix_size),
    do: (2 + ((row - 1) * columns + (columns - suffix_size))) * @element_byte_size

  defp format_row_head_tail(<<>>, _, _), do: <<>>

  defp format_row_head_tail(<<val::binary-4, rest::binary>>, 1, prefix_size)
       when prefix_size > 0 do
    <<format_float(val) <> IO.ANSI.reset() <> " │\n│" <> IO.ANSI.yellow(),
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
  @zero <<0, 0, 0, 0>>

  defp format_float(@not_a_number),
    do: "#{IO.ANSI.red()}#{String.pad_leading("NaN ", @element_chars_size)}#{IO.ANSI.yellow()}"

  defp format_float(@positive_infinity),
    do: "#{IO.ANSI.cyan()}#{String.pad_leading("∞  ", @element_chars_size)}#{IO.ANSI.yellow()}"

  defp format_float(@negative_infinity),
    do: "#{IO.ANSI.cyan()}#{String.pad_leading("-∞  ", @element_chars_size)}#{IO.ANSI.yellow()}"

  defp format_float(@zero),
    do:
      "#{IO.ANSI.color(2, 2, 2)}#{String.pad_leading("0.0", @element_chars_size)}#{
        IO.ANSI.yellow()
      }"

  defp format_float(<<f::float-little-32>>), do: format_float(f)

  defp format_float(f) when is_float(f) do
    f
    |> Float.round(5)
    |> Float.to_string()
    |> String.pad_leading(@element_chars_size)
  end

  def ffloat(f) when is_float(f) do
    af = abs(f)
    frac = af - trunc(af)

    precision =
      cond do
        af >= 10_000 -> 6
        af >= 1_000 -> 5
        af >= 100 -> 4
        af >= 10 -> 3
        frac == 0.0 -> 2
        frac > 0 -> 3
        true -> 3
      end

    :io_lib.format(" ~7.*g", [precision, f]) |> to_string()
  end

  defp insert_vertical_ellipsis_row(
         matrix_as_string,
         _prefix_length,
         _suffix_length,
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
         prefix_length,
         suffix_length,
         _columns,
         _suffix_size,
         _prefix_size
       ) do
    String.replace(
      matrix_as_string,
      ~r/\n.*⋮.*\n/,
      "\n│#{String.pad_leading("", prefix_length + 2, "     ⋮  ")}… #{
        String.pad_trailing("", suffix_length + 1, "     ⋮  ")
      }│\n"
    )
  end

  defp row_length(columns, suffix_size, prefix_size) when suffix_size + prefix_size >= columns,
    do: @element_chars_size * columns + 1

  defp row_length(_columns, suffix_size, prefix_size),
    do: @element_chars_size * (suffix_size + prefix_size) + 5

  defp joiner(columns, suffix_size, prefix_size) when suffix_size + prefix_size >= columns,
    do: " #{IO.ANSI.white()}│\n│#{IO.ANSI.yellow()}"

  defp joiner(_, _, _), do: IO.ANSI.reset() <> "  … " <> IO.ANSI.yellow()

  defp header(%Matrex{} = m), do: header(m[:rows], m[:columns])

  defp header(rows, columns),
    do:
      "#{IO.ANSI.reset()}#Matrex[#{IO.ANSI.yellow()}#{rows}#{IO.ANSI.reset()}×#{IO.ANSI.yellow()}#{
        columns
      }#{IO.ANSI.reset()}]"

  #
  # Heatmap
  #
  @spec heatmap(Matrex.t(), :mono24bit | :color24bit | :mono8 | :color8 | :mono256 | :color256) ::
          Matrex.t()
  def heatmap(%Matrex{} = m, type \\ :mono256, opts \\ []) do
    mn = Matrex.min_finite(m)
    mx = Matrex.max_finite(m)
    range = if mx != mn, do: mx - mn, else: 1

    at_opts = Keyword.take(opts, [:at])
    title = Keyword.get(opts, :title, header(m))

    IO.write("#{hide_cursor()}#{row_prefix(at_opts, 0)}#{title}")
    IO.write(row_prefix(at_opts, 1) <> top_row(m[:cols]))

    n_lines = div(m[:rows], 2) + rem(m[:rows], 2)

    1..n_lines
    |> Enum.map(fn rp ->
      top_row = m[rp * 2 - 1]
      bottom_row = if rp * 2 <= m[:rows], do: m[rp * 2], else: nil
      {rows_pair, _, _} = rows_pair_to_ascii(top_row, bottom_row, mn, range, type)

      <<"#{row_prefix(at_opts, rp + 1)}│", rows_pair::binary, "\e[0m│">>
    end)
    |> Enum.join()
    |> IO.write()

    IO.puts(row_prefix(at_opts, n_lines + 2) <> bottom_row(m[:cols]) <> show_cursor())

    m
  end

  defp hide_cursor(), do: "\e[?25l"
  defp show_cursor(), do: "\e[?25h"

  defp row_prefix([], 0), do: ""
  defp row_prefix([], _), do: "\n"
  defp row_prefix([at: {row, col}], r), do: "\e[#{row + r};#{col}H"

  # If we've got last odd row
  defp rows_pair_to_ascii(top_row, nil, min, range, ttype) do
    1..top_row[:columns]
    |> Enum.reduce({"#{IO.ANSI.inverse()}", "", nil}, fn c, {result, prev_top_pixel_color, nil} ->
      top_pixel_color = val_to_color(ttype, top_row[c], min, range)

      {<<result::binary,
         "#{
           ascii_escape(
             escape_color(ttype, :foreground, top_pixel_color, prev_top_pixel_color),
             ""
           )
         }▄">>, top_pixel_color, nil}
    end)
  end

  defp rows_pair_to_ascii(top_row, bottom_row, min, range, ttype) do
    1..top_row[:columns]
    |> Enum.reduce({"", "", ""}, fn c, {result, prev_top_pixel_color, prev_bottom_pixel_color} ->
      top_pixel_color = val_to_color(ttype, top_row[c], min, range)
      bottom_pixel_color = val_to_color(ttype, bottom_row[c], min, range)

      {<<result::binary,
         pixels(
           ttype,
           top_pixel_color,
           prev_top_pixel_color,
           bottom_pixel_color,
           prev_bottom_pixel_color
         )::binary>>, top_pixel_color, bottom_pixel_color}
    end)
  end

  # Do not set color again, if it's equal to the previous one
  defp escape_color(_, _, color, color), do: ""
  defp escape_color(_, _, nil, _), do: ""

  defp escape_color(ttype, :foreground, color, _prev_color)
       when ttype in [:mono24bit, :color24bit],
       do: "38;2;#{color}"

  defp escape_color(ttype, :background, color, _prev_color)
       when ttype in [:mono24bit, :color24bit],
       do: "48;2;#{color}"

  defp escape_color(ttype, :foreground, color, _prev_color) when ttype in [:mono256, :color256],
    do: "38;5;#{color}"

  defp escape_color(ttype, :background, color, _prev_color) when ttype in [:mono256, :color256],
    do: "48;5;#{color}"

  defp escape_color(ttype, :foreground, color, _prev_color) when ttype in [:mono8, :color8],
    do: "3#{color}"

  defp escape_color(ttype, :background, color, _prev_color) when ttype in [:mono8, :color8],
    do: "4#{color}"

  defp ascii_escape("", ""), do: ""
  defp ascii_escape(color1, ""), do: "\e[#{color1}m"
  defp ascii_escape("", color2), do: "\e[#{color2}m"
  defp ascii_escape(color1, color2), do: "\e[#{color1};#{color2}m"

  defp pixels(
         ttype,
         pixel_color,
         prev_top_pixel_color,
         pixel_color,
         _prev_bottom_pixel_color
       ) do
    "#{ascii_escape(escape_color(ttype, :background, pixel_color, prev_top_pixel_color), "")} "
  end

  defp pixels(
         ttype,
         top_pixel_color,
         prev_pixel_color,
         bottom_pixel_color,
         prev_pixel_color
       ) do
    "#{
      ascii_escape(
        escape_color(ttype, :background, top_pixel_color, prev_pixel_color),
        escape_color(ttype, :foreground, bottom_pixel_color, nil)
      )
    }▄"
  end

  defp pixels(
         ttype,
         top_pixel_color,
         prev_top_pixel_color,
         bottom_pixel_color,
         prev_bottom_pixel_color
       ) do
    "#{
      ascii_escape(
        escape_color(ttype, :background, top_pixel_color, prev_top_pixel_color),
        escape_color(ttype, :foreground, bottom_pixel_color, prev_bottom_pixel_color)
      )
    }▄"
  end

  # Mark float special values on the heatmap
  defp val_to_color(:mono24bit, :nan, _, _), do: "255;0;0"
  defp val_to_color(:color24bit, :nan, _, _), do: "255;255;255"
  defp val_to_color(ttype, :inf, _, _) when ttype in [:mono24bit, :color24bit], do: "0;128;255"
  defp val_to_color(ttype, :neg_inf, _, _) when ttype in [:mono24bit, :color24bit], do: "0;0;255"

  defp val_to_color(:mono256, :nan, _, _), do: "196"
  defp val_to_color(:color256, :nan, _, _), do: "0"
  defp val_to_color(ttype, :inf, _, _) when ttype in [:mono256, :color256], do: "87"
  defp val_to_color(ttype, :neg_inf, _, _) when ttype in [:mono256, :color256], do: "27"

  defp val_to_color(:mono8, :nan, _, _), do: "1"
  defp val_to_color(:color8, :nan, _, _), do: "7"
  defp val_to_color(ttype, :inf, _, _) when ttype in [:mono8, :color8], do: "6"
  defp val_to_color(ttype, :neg_inf, _, _) when ttype in [:mono8, :color8], do: "1"

  defp val_to_color(:mono24bit, val, mn, range) do
    c = trunc((val - mn) * 255 / range)
    "#{c};#{c};#{c}"
  end

  defp val_to_color(:mono256, val, mn, range) do
    case trunc((val - mn) * 25 / range) do
      0 -> "0"
      25 -> "231"
      c -> "#{231 + c}"
    end
  end

  defp val_to_color(:mono8, val, mn, range) do
    c = round((val - mn) / range) |> trunc()
    "#{elem({"0", "7"}, c)}"
  end

  defp val_to_color(:color24bit, val, mn, range) do
    {r, g, b} = val_to_rgb(val, mn, range)
    "#{trunc(r * 255)};#{trunc(g * 255)};#{trunc(b * 255)}"
  end

  defp val_to_color(:color256, val, mn, range) do
    {r, g, b} = val_to_rgb(val, mn, range)
    color = 16 + 36 * trunc(r * 5) + 6 * trunc(g * 5) + trunc(b * 5)
    "#{color}"
  end

  defp val_to_color(:color8, val, mn, range) do
    {r, g, b} = val_to_rgb(val, mn, range)
    color = trunc(round(r)) + 2 * trunc(round(g)) + 4 * trunc(round(b))
    "#{color}"
  end

  @spec val_to_rgb(float, float, float) :: float
  defp val_to_rgb(val, mn, range) do
    # Color points for the heatmap. You can have as many of them, as you wish.
    cps =
      {[r: 0, g: 0, b: 0.3], [r: 0, g: 0, b: 1], [r: 0, g: 0.7, b: 0.7], [r: 0, g: 1, b: 0],
       [r: 1, g: 1, b: 0], [r: 1, g: 0, b: 0]}

    # Normalize value into [0, 1] range.
    vn = (val - mn) / range

    # Find indices of two color points, between wich this value is located (idx1, idx2),
    # and offset from the lower color point (frac).
    {idx1, idx2, frac} =
      cond do
        vn >= 1 ->
          {tuple_size(cps) - 1, tuple_size(cps) - 1, 0.0}

        vn <= 0 ->
          {0, 0, 0.0}

        true ->
          i = vn * (tuple_size(cps) - 1)
          idx1 = trunc(i)
          idx2 = idx1 + 1
          frac = i - idx1

          {idx1, idx2, frac}
      end

    r = (elem(cps, idx2)[:r] - elem(cps, idx1)[:r]) * frac + elem(cps, idx1)[:r]
    g = (elem(cps, idx2)[:g] - elem(cps, idx1)[:g]) * frac + elem(cps, idx1)[:g]
    b = (elem(cps, idx2)[:b] - elem(cps, idx1)[:b]) * frac + elem(cps, idx1)[:b]
    {r, g, b}
  end
end
