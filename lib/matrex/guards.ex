defmodule Matrex.Guards do
  defguard inside_matrex(row, col, rows, columns)
           when row >= 1 and row <= rows and col >= 1 and col <= columns
end
