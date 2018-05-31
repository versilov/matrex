defmodule Matrex.Array do
  @moduledoc """
  NumPy style multidimensional array
  """
  alias Matrex.Array

  @enforce_keys [:data, :type, :shape, :strides]
  defstruct data: nil, type: :float, strides: {}, shape: {}

  @type element :: number | :nan | :inf | :neg_inf
  @type index :: pos_integer
  @type array :: %Array{data: binary, type: atom, shape: tuple, strides: tuple}
  @type t :: array

  def zeros({rows, cols} = shape, type \\ :float) do
    bitsize = rows * cols * sizeof(type)

    %Array{
      data: <<0::size(bitsize)>>,
      type: type,
      shape: shape,
      strides: {cols * sizeof(type), sizeof(type)}
    }
  end

  def sizeof(:float), do: 32
  def sizeof(:double), do: 64
  def sizeof(:byte), do: 8
  def sizeof(:bool), do: 1
end
