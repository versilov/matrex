defmodule Matrex.Guards do
  @moduledoc false
  defmacro inside_matrex(row, col, rows, columns),
    do:
      quote(
        do:
          unquote(row) >= 1 and unquote(row) <= unquote(rows) and unquote(col) >= 1 and
            unquote(col) <= unquote(columns)
      )
end
