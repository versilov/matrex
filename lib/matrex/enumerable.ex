defimpl Enumerable, for: Matrex do
  defmacrop type_and_size() do
    {_type, size} = Module.get_attribute(__CALLER__.module, :type_and_size)

    quote do: binary - unquote(size)
  end

  import Matrex, only: [binary_to_element: 2]

  @doc false
  def count(%Matrex{shape: shape}), do: {:ok, Matrex.elements_count(shape)}

  @doc false
  def member?(%Matrex{} = matrex, element), do: {:ok, Matrex.contains?(matrex, element)}

  @doc false
  def slice(%Matrex{shape: {rows, cols}, data: body, type: type}) do
    {:ok, rows * cols,
     fn start, length ->
       apply(__MODULE__, :"binary_to_list_#{type}", [
         binary_part(
           body,
           start * Matrex.element_size(type),
           length * Matrex.element_size(type)
         )
       ])
     end}
  end

  @doc false
  def reduce(%Matrex{data: body, type: type}, acc, fun),
    do: apply(__MODULE__, :"reduce_each_#{type}", [body, acc, fun])

  types = [
    float64: {:float, 8},
    float32: {:float, 4},
    byte: {:integer, 1},
    int16: {:integer, 2},
    int32: {:integer, 4},
    int64: {:integer, 8}
  ]

  for {guard, type_and_size} <- types do
    @guard guard
    @type_and_size type_and_size

    def unquote(:"reduce_each_#{@guard}")(_, {:halt, acc}, _fun), do: {:halted, acc}

    def unquote(:"reduce_each_#{@guard}")(matrix, {:suspend, acc}, fun),
      do: {:suspended, acc, &unquote(:"reduce_each_#{@guard}")(matrix, &1, fun)}

    def unquote(:"reduce_each_#{@guard}")(
          <<elem::type_and_size(), rest::binary>>,
          {:cont, acc},
          fun
        ),
        do:
          unquote(:"reduce_each_#{@guard}")(
            rest,
            fun.(binary_to_element(elem, @guard), acc),
            fun
          )

    def unquote(:"reduce_each_#{@guard}")(<<>>, {:cont, acc}, _fun), do: {:done, acc}

    def unquote(:"binary_to_list_#{@guard}")(<<elem::type_and_size(), rest::binary>>),
      do: [binary_to_element(elem, @guard) | unquote(:"binary_to_list_#{@guard}")(rest)]

    def unquote(:"binary_to_list_#{@guard}")(<<>>), do: []
  end
end
