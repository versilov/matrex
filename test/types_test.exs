defmodule TypesTest do
  use ExUnit.Case, async: true
  import Matrex

  (Matrex.types() -- [:bool])
  |> Enum.each(fn type1 ->
    (Matrex.types() -- [:bool])
    |> Enum.each(fn type2 ->
      @type1 type1
      @type2 type2

      test "#to_type converts matrex from #{type1} to #{type2}" do
        m = magic(15, @type1)
        m2 = to_type(m, @type2)

        compare =
          m
          |> Matrex.apply(&trunc(&1 - at(m2, &2)))
          |> sum()

        assert compare == 0
        assert to_type(m2, @type1) == m
      end
    end)
  end)
end
