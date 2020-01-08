defmodule Matrex.Guards do
  @moduledoc false
  defmacro inside_matrex(row, col, rows, columns),
    do:
      quote(
        do:
          unquote(row) >= 1 and unquote(row) <= unquote(rows) and unquote(col) >= 1 and
            unquote(col) <= unquote(columns)
      )

  defmacro vector_data(size, body) do
    quote do
      %Matrex{
        data: <<
          <<1, 0, 0, 0>>,
          unquote(size)::unsigned-integer-little-32,
          unquote(body)::binary
        >>
      }
    end
  end

  defmacro vector_data(size, body, data) do
    quote do
      %Matrex{
        data: <<
          <<1, 0, 0, 0>>,
          unquote(size)::unsigned-integer-little-32,
          unquote(body)::binary
        >> = unquote(data)
      }
    end
  end

  defmacro matrex_data(rows, columns, body) do
    quote do
      %Matrex{
        data: <<
          unquote(rows)::unsigned-integer-little-32,
          unquote(columns)::unsigned-integer-little-32,
          unquote(body)::binary
        >>
      }
    end
  end

  defmacro matrex_data(rows, columns, body, data) do
    quote do
      %Matrex{
        data:
          <<
            unquote(rows)::unsigned-integer-little-32,
            unquote(columns)::unsigned-integer-little-32,
            unquote(body)::binary
          >> = unquote(data)
      }
    end
  end

end
