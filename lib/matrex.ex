defmodule Matrex do
  @moduledoc """
  Performs fast operations on matrices using native C code and CBLAS library.

  ## Access behaviour

  Access behaviour is partly implemented for Matrex, so you can do:

  ```elixir

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> m[2][3]
      7.0
  ```
  Or even:
  ```elixir

      iex> m[1..2]
      #Matrex[2×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      └                         ┘
  ```

  There are also several shortcuts for getting dimensions of matrix:
  ```elixir

      iex> m[:rows]
      3

      iex> m[:size]
      {3, 3}
  ```
  calculating maximum value of the whole matrix:
  ```elixir

      iex> m[:max]
      9.0
  ```
  or just one of it's rows:
  ```elixir

      iex> m[2][:max]
      7.0
  ```
  calculating one-based index of the maximum element for the whole matrix:
  ```elixir

      iex> m[:argmax]
      8
  ```
  and a row:
  ```elixir

      iex> m[2][:argmax]
      3
  ```
  ## Inspect protocol

  Matrex implements `Inspect` and looks nice in your console:

  ![Inspect Matrex](https://raw.githubusercontent.com/versilov/matrex/master/docs/matrex_inspect.png)

  ## Math operators overloading

  `Matrex.Operators` module redefines `Kernel` math operators (+, -, *, / <|>) and
  defines some convenience functions, so you can write calculations code in more natural way.

  It should be used with great caution. We suggest using it only inside specific functions
  and only for increased readability, because using `Matrex` module functions, especially
  ones which do two or more operations at one call, are 2-3 times faster.

  ### Example

  ```elixir

      def lr_cost_fun_ops(%Matrex{} = theta, {%Matrex{} = x, %Matrex{} = y, lambda} = _params)
          when is_number(lambda) do
        # Turn off original operators
        import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]
        import Matrex.Operators
        import Matrex

        m = y[:rows]

        h = sigmoid(x * theta)
        l = ones(size(theta)) |> set(1, 1, 0.0)

        j = (-t(y) * log(h) - t(1 - y) * log(1 - h) + lambda / 2 * t(l) * pow2(theta)) / m

        grad = (t(x) * (h - y) + (theta <|> l) * lambda) / m

        {scalar(j), grad}
      end
  ```


  The same function, coded with module methods calls (2.5 times faster):

  ```elixir
      def lr_cost_fun(%Matrex{} = theta, {%Matrex{} = x, %Matrex{} = y, lambda} = _params)
          when is_number(lambda) do
        m = y[:rows]

        h = Matrex.dot_and_apply(x, theta, :sigmoid)
        l = Matrex.ones(theta[:rows], theta[:cols]) |> Matrex.set(1, 1, 0)

        regularization =
          Matrex.dot_tn(l, Matrex.square(theta))
          |> Matrex.scalar()
          |> Kernel.*(lambda / (2 * m))

        j =
          y
          |> Matrex.dot_tn(Matrex.apply(h, :log), -1)
          |> Matrex.subtract(
            Matrex.dot_tn(
              Matrex.subtract(1, y),
              Matrex.apply(Matrex.subtract(1, h), :log)
            )
          )
          |> Matrex.scalar()
          |> (fn
                :nan -> :nan
                x -> x / m + regularization
              end).()

        grad =
          x
          |> Matrex.dot_tn(Matrex.subtract(h, y))
          |> Matrex.add(Matrex.multiply(theta, l), 1.0, lambda)
          |> Matrex.divide(m)

        {j, grad}
      end
  ```

  ## Enumerable protocol

  Matrex implements `Enumerable`, so, all kinds of `Enum` functions are applicable:

  ```elixir

      iex> Enum.member?(m, 2.0)
      true

      iex> Enum.count(m)
      9

      iex> Enum.sum(m)
      45
  ```

  For functions, that exist both in `Enum` and in `Matrex` it's preferred to use Matrex
  version, beacuse it's usually much, much faster. I.e., for 1 000 x 1 000 matrix `Matrex.sum/1`
  and `Matrex.to_list/1` are 438 and 41 times faster, respectively, than their `Enum` counterparts.

  ## Saving and loading matrix

  You can save/load matrix with native binary file format (extra fast)
  and CSV (slow, especially on large matrices).

  Matrex CSV format is compatible with GNU Octave CSV output,
  so you can use it to exchange data between two systems.

  ### Example

  ```elixir

      iex> Matrex.random(5) |> Matrex.save("rand.mtx")
      :ok
      iex> Matrex.load("rand.mtx")
      #Matrex[5×5]
      ┌                                         ┐
      │ 0.05624 0.78819 0.29995 0.25654 0.94082 │
      │ 0.50225 0.22923 0.31941  0.3329 0.78058 │
      │ 0.81769 0.66448 0.97414 0.08146 0.21654 │
      │ 0.33411 0.59648 0.24786 0.27596 0.09082 │
      │ 0.18673 0.18699 0.79753 0.08101 0.47516 │
      └                                         ┘
      iex> Matrex.magic(5) |> Matrex.divide(Matrex.eye(5)) |> Matrex.save("nan.csv")
      :ok
      iex> Matrex.load("nan.csv")
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0     ∞       ∞       ∞       ∞   │
      │     ∞       4.0     ∞       ∞       ∞   │
      │     ∞       ∞      12.0     ∞       ∞   │
      │     ∞       ∞       ∞      25.0     ∞   │
      │     ∞       ∞       ∞       ∞       8.0 │
      └                                         ┘
  ```

  ## NaN and Infinity

  Float special values, like `:nan` and `:inf` live well inside matrices,
  can be loaded from and saved to files.
  But when getting them into Elixir they are transferred to `:nan`,`:inf` and `:neg_inf` atoms,
  because BEAM does not accept special values as valid floats.

  ```elixir
      iex> m = Matrex.eye(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     0.0     0.0 │
      │     0.0     1.0     0.0 │
      │     0.0     0.0     1.0 │
      └                         ┘

      iex> n = Matrex.divide(m, Matrex.zeros(3))
      #Matrex[3×3]
      ┌                         ┐
      │     ∞      NaN     NaN  │
      │    NaN      ∞      NaN  │
      │    NaN     NaN      ∞   │
      └                         ┘

      iex> n[1][1]
      :inf

      iex> n[1][2]
      :nan
  ```

  """

  alias Matrex.NIFs

  @enforce_keys [:data, :shape, :strides, :type]
  defstruct data: nil, type: :float32, strides: {}, shape: {}
  @types [:float32, :float64, :int16, :int32, :int64, :byte, :bool]
  @floats [:float32, :float64]

  @type element :: number | :nan | :inf | :neg_inf
  @type type :: :float32 | :float64 | :int16 | :int32 | :int64 | :byte | :bool
  @type index :: pos_integer
  @type position :: tuple
  @type shape :: tuple
  @type matrex :: %Matrex{data: binary, type: type, shape: tuple, strides: tuple}
  @type t :: matrex

  #############################
  # Util functions
  #############################

  # Available C math functions for floats
  @math_functions [
    :exp,
    :exp2,
    :sigmoid,
    :expm1,
    :log,
    :log2,
    :sqrt,
    :cbrt,
    :ceil,
    :floor,
    :truncate,
    :round,
    :abs,
    :sin,
    :cos,
    :tan,
    :asin,
    :acos,
    :atan,
    :sinh,
    :cosh,
    :tanh,
    :asinh,
    :acosh,
    :atanh,
    :erf,
    :erfc,
    :tgamma,
    :lgamma
  ]

  # Size of matrix element in bytes and bits
  @element_size 4

  @doc false
  def element_size(:float32), do: 4
  def element_size(:float64), do: 8
  def element_size(:int16), do: 2
  def element_size(:int32), do: 4
  def element_size(:int64), do: 8
  def element_size(:byte), do: 1
  # Actually, for bool this is bitsize
  def element_size(:bool), do: 1

  defp element_bit_size(:float32), do: 32
  defp element_bit_size(:float64), do: 64
  defp element_bit_size(:int16), do: 16
  defp element_bit_size(:int32), do: 32
  defp element_bit_size(:int64), do: 64
  defp element_bit_size(:byte), do: 8
  defp element_bit_size(:bool), do: 1

  # Float special values in binary form
  @not_a_number_float32 <<0, 0, 192, 255>>
  @positive_infinity_float32 <<0, 0, 128, 127>>
  @negative_infinity_float32 <<0, 0, 128, 255>>

  @not_a_number_float64 <<0, 0, 0, 0, 0, 0, 248, 255>>
  @positive_infinity_float64 <<0, 0, 0, 0, 0, 0, 240, 127>>
  @negative_infinity_float64 <<0, 0, 0, 0, 0, 0, 240, 255>>

  @doc false
  @spec binary_to_float32(<<_::32>>) :: element
  def binary_to_float32(@not_a_number_float32), do: :nan
  def binary_to_float32(@positive_infinity_float32), do: :inf
  def binary_to_float32(@negative_infinity_float32), do: :neg_inf
  def binary_to_float32(<<val::float-little-32>>), do: val

  # To be deleted
  defdelegate binary_to_float(bin), to: __MODULE__, as: :binary_to_float32

  @spec binary_to_float64(<<_::64>>) :: element
  def binary_to_float64(@not_a_number_float64), do: :nan
  def binary_to_float64(@positive_infinity_float64), do: :inf
  def binary_to_float64(@negative_infinity_float64), do: :neg_inf
  def binary_to_float64(<<val::float-little-64>>), do: val

  def binary_to_int16(<<int::integer-little-16>>), do: int
  def binary_to_int32(<<int::integer-little-32>>), do: int
  def binary_to_int64(<<int::integer-little-64>>), do: int
  def binary_to_byte(<<byte>>), do: byte
  def binary_to_bool(<<b::size(1)>>), do: b

  @spec float32_to_binary(element) :: <<_::32>>
  def float32_to_binary(val) when is_number(val), do: <<val::float-little-32>>
  def float32_to_binary(:nan), do: @not_a_number_float32
  def float32_to_binary(:inf), do: @positive_infinity_float32
  def float32_to_binary(:neg_inf), do: @negative_infinity_float32

  def float32_to_binary(unknown_val),
    do: raise(ArgumentError, message: "Unknown matrix element value: #{unknown_val}")

  # To be deleted
  defdelegate float_to_binary(float), to: __MODULE__, as: :float32_to_binary

  @spec float64_to_binary(element) :: <<_::64>>
  defp float64_to_binary(val) when is_number(val), do: <<val::float-little-64>>
  defp float64_to_binary(:nan), do: @not_a_number_float64
  defp float64_to_binary(:inf), do: @positive_infinity_float64
  defp float64_to_binary(:neg_inf), do: @negative_infinity_float64

  defp float64_to_binary(unknown_val),
    do: raise(ArgumentError, message: "Unknown matrix element value: #{unknown_val}")

  @spec element_to_binary(element, type) :: binary
  defp element_to_binary(elem, :byte), do: <<elem::unsigned-integer-8>>
  defp element_to_binary(elem, :int16), do: <<elem::integer-little-16>>
  defp element_to_binary(elem, :int32), do: <<elem::integer-little-32>>
  defp element_to_binary(elem, :int64), do: <<elem::integer-little-64>>

  defp element_to_binary(:nan, :float32), do: @not_a_number_float32
  defp element_to_binary(:inf, :float32), do: @positive_infinity_float32
  defp element_to_binary(:neg_inf, :float32), do: @negative_infinity_float32
  defp element_to_binary(elem, :float32), do: <<elem::float-little-32>>

  defp element_to_binary(:nan, :float64), do: @not_a_number_float64
  defp element_to_binary(:inf, :float64), do: @positive_infinity_float64
  defp element_to_binary(:neg_inf, :float64), do: @negative_infinity_float64
  defp element_to_binary(elem, :float64), do: <<elem::float-little-64>>

  @spec binary_to_element(binary, type) :: element
  defp binary_to_element(<<elem::unsigned-integer-8>>, :byte), do: elem
  defp binary_to_element(<<elem::integer-little-16>>, :int16), do: elem
  defp binary_to_element(<<elem::integer-little-32>>, :int32), do: elem
  defp binary_to_element(<<elem::integer-little-64>>, :int64), do: elem

  defp binary_to_element(@not_a_number_float32, :float32), do: :nan
  defp binary_to_element(@positive_infinity_float32, :float32), do: :inf
  defp binary_to_element(@negative_infinity_float32, :float32), do: :neg_inf
  defp binary_to_element(<<elem::float-little-32>>, :float32), do: elem

  defp binary_to_element(@not_a_number_float64, :float64), do: :nan
  defp binary_to_element(@positive_infinity_float64, :float64), do: :inf
  defp binary_to_element(@negative_infinity_float64, :float64), do: :neg_inf
  defp binary_to_element(<<elem::float-little-64>>, :float64), do: elem

  @doc false
  def strides({_}, type), do: {element_size(type)}

  def strides(shape, type) when is_tuple(shape) and is_atom(type) do
    Enum.reduce((tuple_size(shape) - 1)..1, {{element_size(type)}, 1}, fn i, {strides, acc} ->
      stride = acc * elem(shape, i)
      {Tuple.insert_at(strides, 0, stride * element_size(type)), stride}
    end)
    |> elem(0)
  end

  # Offset of the element with position `pos` in bytes. Zero based.
  defp offset(strides, pos) when tuple_size(strides) == tuple_size(pos),
    do:
      Enum.reduce(0..(tuple_size(pos) - 1), 0, fn i, off ->
        elem(strides, i) * (elem(pos, i) - 1) + off
      end)

  # Calculate next position tuple for the given position and shape
  defp next_pos({r, c}, {rows, cols}), do: if(c + 1 > cols, do: {r + 1, 1}, else: {r, c + 1})

  defp next_pos(pos, shape) do
    Enum.reduce_while((tuple_size(shape) - 1)..0, pos, fn i, p ->
      new_coord = elem(p, i) + 1

      if new_coord <= elem(shape, i) do
        {:halt, put_elem(p, i, new_coord)}
      else
        {:cont, put_elem(p, i, 1)}
      end
    end)
  end

  defp inside_shape(pos, shape) do
    Enum.zip(Tuple.to_list(pos), Tuple.to_list(shape))
    |> Enum.all?(fn {p, s} -> p >= 1 && p <= s end)
  end

  @doc """
  Count elements of a matirx.

  ## Example

      iex> Matrex.random(5, 6) |> elements_count()
      30

  """
  def elements_count(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(&(&1 * &2))
  end

  defp byte_size(shape, type) when is_tuple(shape) and type in @types,
    do: elements_count(shape) * element_size(type)

  @compile {:inline,
            add: 2,
            argmax: 1,
            at: 3,
            binary_to_float32: 1,
            binary_to_float64: 1,
            binary_to_int16: 1,
            binary_to_int32: 1,
            binary_to_int64: 1,
            binary_to_byte: 1,
            binary_to_bool: 1,
            call_nif: 3,
            column_to_list: 2,
            contains?: 2,
            divide: 2,
            dot: 3,
            dot_and_add: 3,
            dot_nt: 2,
            dot_tn: 2,
            eye: 1,
            element_to_string: 1,
            element_to_binary: 2,
            element_size: 1,
            fill: 3,
            fill: 2,
            first: 1,
            fetch: 2,
            float_to_binary: 1,
            max: 1,
            multiply: 2,
            ones: 2,
            ones: 1,
            parse_float: 1,
            random: 2,
            random: 1,
            reshape: 3,
            row_to_list: 2,
            row: 2,
            set: 4,
            size: 1,
            square: 1,
            subtract: 2,
            subtract_inverse: 2,
            sum: 1,
            to_list: 1,
            to_list_of_lists: 1,
            to_row: 1,
            to_column: 1,
            transpose: 1,
            update: 3,
            zeros: 2,
            zeros: 1}

  @behaviour Access

  defmacrop type_and_size() do
    {type, size} = Module.get_attribute(__CALLER__.module, :type_and_size)

    if size == 8 do
      quote do: size(unquote(size))
    else
      quote do: size(unquote(size)) - unquote(type)() - little
    end
  end

  defmacrop binary_size() do
    {type, size} = Module.get_attribute(__CALLER__.module, :type_and_size)

    quote do: binary - unquote(div(size, 8))
  end

  @spec call_nif(atom, type, list) :: any
  defp call_nif(func, type, args), do: Kernel.apply(NIFs, :"#{func}_#{type}", args)

  @spec call_typed(atom, type, list) :: any
  defp call_typed(func, type, args), do: Kernel.apply(__MODULE__, :"#{func}_#{type}", args)

  @impl Access
  def fetch(matrex, key)

  # Single dimension matrex
  def fetch(%Matrex{shape: {_}} = matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, at(matrex, {key})}

  # Horizontal vector
  def fetch(%Matrex{shape: {1, _}} = matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, at(matrex, {1, key})}

  # Vertical vector
  def fetch(%Matrex{shape: {_, 1}} = matrex, key)
      when is_integer(key) and key > 0,
      do: {:ok, at(matrex, {key, 1})}

  # TODO: Return submatrix
  # def fetch(matrex, key)
  #     when is_integer(key) and key > 0,
  #     do: {:ok, row(matrex, key)}

  # Slice on horizontal vector
  def fetch(%Matrex{shape: {1, columns}, data: data, type: type} = matrex, a..b)
      when b > a and a > 0 and b <= columns do
    data = binary_part(data, (a - 1) * element_size(type), (b - a + 1) * element_size(type))
    {:ok, %{matrex | shape: {1, b - a + 1}, data: data}}
  end

  def fetch(%Matrex{shape: {rows, columns}, data: data, type: type} = matrex, a..b)
      when b > a and a > 0 and b <= rows do
    data =
      binary_part(
        data,
        (a - 1) * columns * element_size(type),
        (b - a + 1) * columns * element_size(type)
      )

    {:ok, %{matrex | shape: {b - a + 1, columns}, data: data}}
  end

  def fetch(%Matrex{shape: {rows, _}}, :rows), do: {:ok, rows}
  def fetch(%Matrex{shape: {_, cols}}, :cols), do: {:ok, cols}
  def fetch(%Matrex{shape: {_, cols}}, :columns), do: {:ok, cols}
  def fetch(%Matrex{shape: {rows, cols}}, :size), do: {:ok, {rows, cols}}
  def fetch(matrex, :sum), do: {:ok, sum(matrex)}
  def fetch(matrex, :max), do: {:ok, max(matrex)}
  def fetch(matrex, :min), do: {:ok, min(matrex)}
  def fetch(matrex, :argmax), do: {:ok, argmax(matrex)}

  @impl Access
  def pop(%Matrex{shape: {rows, columns}, data: body, type: type}, row)
      when is_integer(row) and row >= 1 and row <= rows do
    get = %Matrex{
      data:
        binary_part(body, (row - 1) * columns * element_size(type), columns * element_size(type)),
      shape: {1, columns},
      strides: strides({1, columns}, type),
      type: type
    }

    update = %Matrex{
      data:
        binary_part(body, 0, (row - 1) * columns * element_size(type)) <>
          binary_part(
            body,
            row * columns * element_size(type),
            (rows - row) * columns * element_size(type)
          ),
      shape: {rows - 1, columns},
      strides: strides({rows - 1, columns}, type),
      type: type
    }

    {get, update}
  end

  # Row out of range
  def pop(%Matrex{} = matrex, _), do: {nil, matrex}

  # To silence warnings
  @impl Access
  def get_and_update(%Matrex{}, _row, _fun), do: raise("not implemented")

  defimpl Inspect do
    @doc false
    def inspect(%Matrex{} = matrex, opts) do
      columns =
        case opts.width do
          :infinity -> 80
          width -> width
        end

      Matrex.Inspect.do_inspect(matrex, columns, 21)
    end
  end

  @doc """
  Adds scalar to matrix.

  See `Matrex.add/4` for details.
  """
  @spec add(matrex, number) :: matrex
  @spec add(number, matrex) :: matrex
  def add(a, b)

  @doc """
  Adds two matrices or scalar to each element of matrix. NIF.

  Can optionally scale any of the two matrices.

  C = αA + βB

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.add(Matrex.new([[1,2,3],[4,5,6]]), Matrex.new([[7,8,9],[10,11,12]]))
      #Matrex[2×3]
      ┌                         ┐
      │     8.0    10.0    12.0 │
      │    14.0    16.0    18.0 │
      └                         ┘

  Adding with scalar:

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.add(m, 1)
      #Matrex[3×3]
      ┌                         ┐
      │     9.0     2.0     7.0 │
      │     4.0     6.0     8.0 │
      │     5.0    10.0     3.0 │
      └                         ┘

  With scaling each matrix:

      iex> Matrex.add(Matrex.new("1 2 3; 4 5 6"), Matrex.new("3 2 1; 6 5 4"), 2.0, 3.0)
      #Matrex[2×3]
      ┌                         ┐
      │     11.0    10.0    9.0 │
      │     26.0    25.0   24.0 │
      └                         ┘
  """

  @spec add(matrex | number, matrex | number, number, number) :: matrex
  def add(a, b, alpha \\ 1.0, beta \\ 1.0)

  def add(%Matrex{data: data, type: type} = matrex, scalar, alpha, _beta)
      when is_number(scalar),
      do: %{matrex | data: call_nif(:add_scalar, type, [data, scalar, alpha])}

  def add(scalar, %Matrex{data: data, type: type} = matrex, alpha, _beta)
      when is_number(scalar),
      do: %{matrex | data: call_nif(:add_scalar, type, [data, scalar, alpha])}

  def add(
        %Matrex{
          data: data1,
          shape: shape,
          strides: strides,
          type: type
        } = a,
        %Matrex{
          data: data2,
          shape: shape,
          strides: strides,
          type: type
        },
        alpha,
        beta
      )
      when is_number(alpha) and is_number(beta),
      do: %{a | data: call_nif(:add, type, [data1, data2, alpha, beta])}

  @doc """

  Applies given function to each element of the matrix and returns the matrex of results. NIF.

  If second argument is an atom, then applies C language math function.

  ## Example

      iex> Matrex.magic(5) |> Matrex.apply(:sigmoid)
      #Matrex[5×5]
      ┌                                         ┐
      │-0.95766-0.53283 0.28366  0.7539 0.13674 │
      │-0.99996-0.65364 0.96017 0.90745 0.40808 │
      │-0.98999-0.83907 0.84385  0.9887-0.54773 │
      │-0.91113 0.00443 0.66032  0.9912-0.41615 │
      │-0.75969-0.27516 0.42418  0.5403 -0.1455 │
      └                                         ┘

  The following math functions from C <math.h> are supported, and also a sigmoid function:

  ```elixir
    :exp, :exp2, :sigmoid, :expm1, :log, :log2, :sqrt, :cbrt, :ceil, :floor, :truncate, :round,
    :abs, :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
    :erf, :erfc, :tgamma, :lgamm
  ```


  If second argument is a function that takes one argument,
  then this function receives the element of the matrix.

  ## Example

      iex> Matrex.magic(5) |> Matrex.apply(&:math.cos/1)
      #Matrex[5×5]
      ┌                                         ┐
      │-0.95766-0.53283 0.28366  0.7539 0.13674 │
      │-0.99996-0.65364 0.96017 0.90745 0.40808 │
      │-0.98999-0.83907 0.84385  0.9887-0.54773 │
      │-0.91113 0.00443 0.66032  0.9912-0.41615 │
      │-0.75969-0.27516 0.42418  0.5403 -0.1455 │
      └                                         ┘


  If second argument is a function that takes two arguments,
  then this function receives the element of the matrix and its one-based index.


  ## Example

      iex> Matrex.ones(5) |> Matrex.apply(fn val, index -> val + index end)
      #Matrex[5×5]
      ┌                                         ┐
      │     2.0     3.0     4.0     5.0     6.0 │
      │     7.0     8.0     9.0    10.0    11.0 │
      │    12.0    13.0    14.0    15.0    16.0 │
      │    17.0    18.0    19.0    20.0    21.0 │
      │    22.0    23.0    24.0    25.0    26.0 │
      └                                         ┘

  If second argument is a function that takes three arguments,
  then this function receives the element of the matrix one-based row index and one-based
  column index of the element.

  ## Example

      iex> Matrex.ones(5) |> Matrex.apply(fn val, row, col -> val + row + col end)
      #Matrex[5×5]
      ┌                                         ┐
      │     3.0     4.0     5.0     6.0     7.0 │
      │     4.0     5.0     6.0     7.0     8.0 │
      │     5.0     6.0     7.0     8.0     9.0 │
      │     6.0     7.0     8.0     9.0    10.0 │
      │     7.0     8.0     9.0    10.0    11.0 │
      └                                         ┘

  """
  @spec apply(
          matrex,
          atom
          | (element -> element)
          | (element, index -> element)
          | (element, index, index -> element)
        ) :: matrex

  def apply(%Matrex{data: data, type: type} = matrex, function_atom)
      when function_atom in @math_functions and type in @floats,
      do: %{matrex | data: call_nif(:apply_math, type, [data, function_atom])}

  def apply(%Matrex{data: data, type: type} = matrex, function) when is_function(function, 1),
    do: %{matrex | data: call_typed(:apply_on_matrix, type, [data, function, <<>>])}

  def apply(%Matrex{data: data, shape: shape, type: type} = matrex, function)
      when is_function(function, 2),
      do: %{
        matrex
        | data:
            call_typed(:apply_on_matrix, type, [
              data,
              function,
              Tuple.duplicate(1, tuple_size(shape)),
              shape,
              <<>>
            ])
      }

  @doc """
  Applies function to elements of two matrices and returns matrix of function results.

  Matrices must be of the same size.

  ## Example

      iex(11)> Matrex.apply(Matrex.random(5), Matrex.random(5), fn x1, x2 -> min(x1, x2) end)
      #Matrex[5×5]
      ┌                                         ┐
      │ 0.02025 0.15055 0.69177 0.08159 0.07237 │
      │ 0.03252 0.14805 0.03627  0.1733 0.58721 │
      │ 0.10865 0.49192 0.12166  0.0573 0.66522 │
      │ 0.13642 0.23838 0.14403 0.57151 0.12359 │
      │ 0.12877 0.12745 0.10933 0.27281 0.35957 │
      └                                         ┘
  """
  @spec apply(matrex, matrex, (element, element -> element)) :: matrex
  def apply(
        %Matrex{data: data1, shape: shape, strides: strides, type: type} = matrex,
        %Matrex{data: data2, shape: shape, strides: strides, type: type},
        function
      )
      when is_function(function, 2) do
    %{
      matrex
      | data: call_typed(:apply_on_matrices, type, [data1, data2, function, <<>>])
    }
  end

  @doc """
  Returns one-based index of the biggest element. NIF.

  There is also `matrex[:argmax]` shortcut for this function.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.argmax(m)
      7

  """
  @spec argmax(matrex) :: index
  def argmax(%Matrex{data: data, type: type}), do: call_nif(:argmax, type, [data]) + 1

  @doc """
  Get element of a matrix at given one-based (row, column) position.

  Negative or out of bound indices will raise an exception.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.at(m, 3, 2)
      9.0
      iex> Matrex.at(m, {3, 1})
      4.0

  You can use `Access` behaviour square brackets for the same purpose,
  but it will be slower:

      iex> m[3][2]
      9.0

  """
  @spec at(matrex, index, index) :: element
  def at(%Matrex{} = matrex, row, col), do: at(matrex, {row, col})

  @spec at(matrex, position) :: element
  def at(matrex, pos)

  @doc """
  Get column of matrix as matrix (vector) in matrex form. One-based.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.column(m, 2)
      #Matrex[3×1]
      ┌         ┐
      │     1.0 │
      │     5.0 │
      │     9.0 │
      └         ┘

  """
  @spec column(matrex, index) :: matrex
  def column(%Matrex{data: data, shape: {rows, columns}, type: type}, col)
      when is_integer(col) and col > 0 and col <= columns do
    data =
      Enum.map(0..(rows - 1), fn row ->
        binary_part(
          data,
          (row * columns + (col - 1)) * element_size(type),
          element_size(type)
        )
      end)
      |> IO.iodata_to_binary()

    %Matrex{
      data: data,
      shape: {rows, 1},
      strides: strides({rows, 1}, type),
      type: type
    }
  end

  @doc """
  Get column of matrix as list of floats. One-based, NIF.


  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.column_to_list(m, 3)
      [6.0, 7.0, 2.0]

  """
  @spec column_to_list(matrex, index) :: [element]
  def column_to_list(%Matrex{data: data, shape: {_rows, columns}, type: type}, column)
      when is_integer(column) and column > 0 and column <= columns,
      do: call_nif(:column_to_list, type, [data, columns, column - 1])

  @doc """
  Concatenate list of matrices along columns.

  The number of rows must be equal.

  ## Example

      iex> Matrex.concat([Matrex.fill(2, 0), Matrex.fill(2, 1), Matrex.fill(2, 2)])                #Matrex[2×6]
      ┌                                                 ┐
      │     0.0     0.0     1.0     1.0     2.0     2.0 │
      │     0.0     0.0     1.0     1.0     2.0     2.0 │
      └                                                 ┘

  """
  @spec concat([matrex]) :: matrex
  def concat([%Matrex{} | _] = list_of_ma), do: Enum.reduce(list_of_ma, &Matrex.concat(&2, &1))

  @doc """
  Concatenate two matrices along rows or columns. NIF.

  The number of rows or columns must be equal.

  ## Examples

      iex> m1 = Matrex.new([[1, 2, 3], [4, 5, 6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> m2 = Matrex.new([[7, 8, 9], [10, 11, 12]])
      #Matrex[2×3]
      ┌                         ┐
      │     7.0     8.0     9.0 │
      │    10.0    11.0    12.0 │
      └                         ┘
      iex> Matrex.concat(m1, m2)
      #Matrex[2×6]
      ┌                                                 ┐
      │     1.0     2.0     3.0     7.0     8.0     9.0 │
      │     4.0     5.0     6.0    10.0    11.0    12.0 │
      └                                                 ┘
      iex> Matrex.concat(m1, m2, :rows)
      #Matrex[4×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      │    10.0    11.0    12.0 │
      └                         ┘
  """
  @spec concat(matrex, matrex, :columns | :rows) :: matrex
  def concat(matrex1, matrex2, axis \\ :columns)

  def concat(
        %Matrex{
          data: data1,
          shape: {rows, _},
          type: type,
          strides: strides
        } = matrex,
        %Matrex{
          data: data2,
          shape: {rows, _},
          type: type,
          strides: strides
        },
        :columns
      ),
      do: %{matrex | data: call_nif(:concat_columns, type, [data1, data2])}

  def concat(
        %Matrex{data: data1, shape: {rows1, columns}, type: type, strides: strides} = matrex,
        %Matrex{data: data2, shape: {rows2, columns}, type: type, strides: strides},
        :rows
      ),
      do: %{matrex | data: data1 <> data2, shape: {rows1 + rows2, columns}}

  def concat(%Matrex{shape: {rows1, columns1}}, %Matrex{shape: {rows2, columns2}}, axis),
    do:
      raise(
        ArgumentError,
        "Cannot concat: #{rows1}×#{columns1} does not fit with #{rows2}×#{columns2} along #{axis}."
      )

  @doc """
  Checks if given element exists in the matrix.

  ## Example

      iex> m = Matrex.new("1 NaN 3; Inf 10 23")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0    NaN      3.0 │
      │     ∞      10.0    23.0 │
      └                         ┘
      iex> Matrex.contains?(m, 1.0)
      true
      iex> Matrex.contains?(m, :nan)
      true
      iex> Matrex.contains?(m, 9)
      false
  """
  @spec contains?(matrex, element) :: boolean
  def contains?(%Matrex{} = matrex, value), do: find(matrex, value) != nil

  @doc """
  Divides two matrices element-wise or matrix by scalar or scalar by matrix. NIF through `find/2`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[10, 20, 25], [8, 9, 4]])
      ...> |> Matrex.divide(Matrex.new([[5, 10, 5], [4, 3, 4]]))
      #Matrex[2×3]
      ┌                         ┐
      │     2.0     2.0     5.0 │
      │     2.0     3.0     1.0 │
      └                         ┘

      iex> Matrex.new([[10, 20, 25], [8, 9, 4]])
      ...> |> Matrex.divide(2)
      #Matrex[2×3]
      ┌                         ┐
      │     5.0    10.0    12.5 │
      │     4.0     4.5     2.0 │
      └                         ┘

      iex> Matrex.divide(100, Matrex.new([[10, 20, 25], [8, 16, 4]]))
      #Matrex[2×3]
      ┌                         ┐
      │    10.0     5.0     4.0 │
      │    12.5    6.25    25.0 │
      └                         ┘

  """
  @spec divide(matrex, number) :: matrex
  @spec divide(number, matrex) :: matrex

  def divide(%Matrex{data: data, type: type} = matrex, scalar) when is_number(scalar),
    do: %{matrex | data: call_nif(:divide_by_scalar, type, [data, scalar])}

  def divide(scalar, %Matrex{data: data, type: type} = matrex) when is_number(scalar),
    do: %{matrex | data: call_nif(:divide_scalar, type, [scalar, data])}

  @spec divide(matrex, matrex, number) :: matrex
  def divide(
        %Matrex{data: dividend, shape: shape, strides: strides, type: type} = matrex,
        %Matrex{
          data: divisor,
          shape: shape,
          strides: strides,
          type: type
        },
        alpha \\ 1.0
      )
      when is_number(alpha),
      do: %{matrex | data: call_nif(:divide, type, [dividend, divisor, alpha])}

  @doc """
  Matrix multiplication. NIF, via `cblas_sgemm()`.

  Number of columns of the first matrix must be equal to the number of rows of the second matrix.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot(Matrex.new([[1, 2], [3, 4], [5, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot(matrex, matrex, number) :: matrex
  def dot(
        %Matrex{data: data1, shape: {rows, common_dim}, type: type},
        %Matrex{
          data: data2,
          shape: {common_dim, columns},
          type: type
        },
        alpha \\ 1.0
      )
      when is_number(alpha),
      do: %Matrex{
        data: call_nif(:dot, type, [data1, data2, rows, common_dim, columns, alpha]),
        shape: {rows, columns},
        strides: strides({rows, columns}, type),
        type: type
      }

  @doc """
  Matrix multiplication with addition of third matrix.  NIF, via `cblas_sgemm()`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_and_add(Matrex.new([[1, 2], [3, 4], [5, 6]]), Matrex.new([[1, 2], [3, 4]]))
      #Matrex[2×2]
      ┌                 ┐
      │    23.0    30.0 │
      │    52.0    68.0 │
      └                 ┘

  """
  @spec dot_and_add(matrex, matrex, matrex) :: matrex
  def dot_and_add(
        %Matrex{data: data1, shape: {rows, common_dim}, type: type},
        %Matrex{data: data2, shape: {common_dim, columns}, type: type},
        %Matrex{data: data3, shape: {rows, columns}, type: type},
        alpha \\ 1.0
      )
      when is_number(alpha),
      do: %Matrex{
        data:
          call_nif(:dot_and_add, type, [
            data1,
            data2,
            rows,
            common_dim,
            columns,
            data3,
            alpha
          ]),
        shape: {rows, columns},
        strides: strides({rows, columns}, type),
        type: type
      }

  @doc """
  Computes dot product of two matrices, then applies math function to each element
  of the resulting matrix.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_and_apply(Matrex.new([[1, 2], [3, 4], [5, 6]]), :sqrt)
      #Matrex[2×2]
      ┌                 ┐
      │ 4.69042  5.2915 │
      │     7.0     8.0 │
      └                 ┘
  """
  @spec dot_and_apply(matrex, matrex, atom) :: matrex
  def dot_and_apply(
        %Matrex{data: data1, shape: {rows, common_dim}, type: type},
        %Matrex{data: data2, shape: {common_dim, columns}, type: type},
        function_atom,
        alpha \\ 1.0
      )
      when function_atom in @math_functions and is_number(alpha) and type in @floats,
      do: %Matrex{
        data:
          call_nif(:dot_and_apply, type, [
            data1,
            data2,
            rows,
            common_dim,
            columns,
            function_atom,
            alpha
          ]),
        shape: {rows, columns},
        strides: strides({rows, columns}, type),
        type: type
      }

  @doc """
  Matrix multiplication where the second matrix needs to be transposed.  NIF, via `cblas_sgemm()`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.dot_nt(Matrex.new([[1, 3, 5], [2, 4, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot_nt(matrex, matrex) :: matrex
  def dot_nt(
        %Matrex{data: data1, shape: {rows, common_dim}, type: type},
        %Matrex{data: data2, shape: {columns, common_dim}, type: type},
        alpha \\ 1.0
      )
      when is_number(alpha),
      do: %Matrex{
        data:
          call_nif(:dot_nt, type, [
            data1,
            data2,
            rows,
            common_dim,
            columns,
            alpha
          ]),
        shape: {rows, columns},
        strides: strides({rows, columns}, type),
        type: type
      }

  @doc """
  Matrix dot multiplication where the first matrix needs to be transposed.  NIF, via `cblas_sgemm()`.

  The result is multiplied by scalar `alpha`.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 4], [2, 5], [3, 6]]) |>
      ...> Matrex.dot_tn(Matrex.new([[1, 2], [3, 4], [5, 6]]))
      #Matrex[2×2]
      ┌                 ┐
      │    22.0    28.0 │
      │    49.0    64.0 │
      └                 ┘

  """
  @spec dot_tn(matrex, matrex, number) :: matrex
  def dot_tn(
        %Matrex{data: data1, shape: {common_dim, rows}, type: type},
        %Matrex{data: data2, shape: {common_dim, columns}, type: type},
        alpha \\ 1.0
      )
      when is_number(alpha),
      do: %Matrex{
        data:
          call_nif(:dot_tn, type, [
            data1,
            data2,
            rows,
            common_dim,
            columns,
            alpha
          ]),
        shape: {rows, columns},
        strides: strides({rows, columns}, type),
        type: type
      }

  @doc """
  Create eye (identity) square matrix of given size.

  ## Examples

      iex> Matrex.eye(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     0.0     0.0 │
      │     0.0     1.0     0.0 │
      │     0.0     0.0     1.0 │
      └                         ┘

      iex> Matrex.eye(3, 2.95)
      #Matrex[3×3]
      ┌                         ┐
      │    2.95     0.0     0.0 │
      │     0.0    2.95     0.0 │
      │     0.0     0.0    2.95 │
      └                         ┘
  """
  @spec eye(index, element) :: matrex

  def eye(size, value \\ 1.0, type \\ :float32)
      when is_integer(size) and is_number(value) and type in @types,
      do: %Matrex{
        data: call_nif(:eye, type, [size, value]),
        shape: {size, size},
        strides: strides({size, size}, type),
        type: type
      }

  @doc """
  Create matrix filled with given value. NIF.

  ## Example

      iex> Matrex.fill(4,3, 55)
      #Matrex[4×3]
      ┌                         ┐
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      └                         ┘
  """
  @spec fill(shape | index, element, type) :: matrex
  def fill(shape, value, type \\ :float32)

  def fill(shape, value, type)
      when (is_tuple(shape) and is_number(value)) or (is_atom(value) and is_atom(type)),
      do: %Matrex{
        data:
          call_nif(:fill, type, [
            elements_count(shape),
            element_to_binary(value, type)
          ]),
        shape: shape,
        strides: strides(shape, type),
        type: type
      }

  @doc """
  Create square matrix filled with given value. Inlined.

  ## Example

      iex> Matrex.fill(3, 55)
      #Matrex[3×3]
      ┌                         ┐
      │    33.0    33.0    33.0 │
      │    33.0    33.0    33.0 │
      │    33.0    33.0    33.0 │
      └                         ┘
  """
  def fill(size, value, type) when is_integer(size), do: fill({size, size}, value, type)

  @doc """
  Find position of the first occurence of the given value in the matrix. NIF.

  Returns {row, column} tuple or nil, if nothing was found. One-based.

  ## Example


  """
  @spec find(matrex, element) :: position | nil
  def find(%Matrex{data: data, type: type}, value)
      when is_number(value) or value in [:nan, :inf, :neg_inf],
      do:
        call_nif(:find, type, [
          data,
          element_to_binary(value, type)
        ])

  @doc """
  Return first element of a matrix.

  ## Example

      iex> Matrex.new([[6,5,4],[3,2,1]]) |> Matrex.first()
      6.0

  """
  @spec first(matrex) :: element
  def first(matrex)

  @doc """
  Prints monochrome or color heatmap of the matrix to the console.

  Supports 8, 256 and 16mln of colors terminals. Monochrome on 256 color palette is the default.

  `type` can be `:mono8`, `:color8`, `:mono256`, `:color256`, `:mono24bit` and `:color24bit`.

  Special float values, like infinity and not-a-number are marked with contrast colors on the map.

  ## Options

    * `:at` — positions heatmap at the specified `{row, col}` position inside terminal.
    * `:title` — sets the title of the heatmap.

  ## Examples

  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/mnist8.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/mnist_sum.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/magic_square.png" width="200px" />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/hot_boobs.png" width="220px"  />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/neurons_mono.png" width="233px"  />&nbsp;
  <img src="https://raw.githubusercontent.com/versilov/matrex/master/docs/logistic_regression.gif" width="180px" />&nbsp;

  """
  @spec heatmap(
          matrex,
          :mono8 | :color8 | :mono256 | :color256 | :mono24bit | :color24bit,
          keyword
        ) :: matrex
  defdelegate heatmap(matrex, type \\ :mono256, opts \\ []), to: Matrex.Inspect

  @doc """
  An alias for `eye/1`.
  """
  @spec identity(index) :: matrex
  defdelegate identity(size), to: __MODULE__, as: :eye

  @doc """
  Returns list of all rows of a matrix as single-row matrices.

  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[6×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘
      iex> Matrex.list_of_rows(m)
      [#Matrex[1×2]
      ┌                 ┐
      │     1.0     2.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     3.0     4.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     5.0     6.0 │
      └                 ┘]


  """
  @spec list_of_rows(matrex) :: [matrex]
  def list_of_rows(%Matrex{data: data, shape: {rows, columns}, type: type}),
    do: do_list_rows(data, rows, columns, type)

  @doc """
  Returns range of rows of a matrix as list of 1-row matrices.

  ## Example

      iex> m = Matrex.reshape(1..12, 6, 2)
      #Matrex[6×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      │     7.0     8.0 │
      │     9.0    10.0 │
      │    11.0    12.0 │
      └                 ┘
      iex> Matrex.list_of_rows(m, 2..4)
      [#Matrex[1×2]
      ┌                 ┐
      │     3.0     4.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     5.0     6.0 │
      └                 ┘,
      #Matrex[1×2]
      ┌                 ┐
      │     7.0     8.0 │
      └                 ┘]

  """
  @spec list_of_rows(matrex, Range.t()) :: [matrex]
  def list_of_rows(%Matrex{data: data, shape: {rows, columns}, type: type}, from..to)
      when from <= to and to <= rows do
    part =
      binary_part(
        data,
        (from - 1) * columns * element_size(type),
        (to - from + 1) * columns * element_size(type)
      )

    do_list_rows(part, to - from + 1, columns, type)
  end

  defp do_list_rows(<<>>, 0, _, _), do: []

  defp do_list_rows(<<rows::binary>>, row_num, columns, type) do
    [
      %Matrex{
        data: binary_part(rows, 0, columns * element_size(type)),
        shape: {1, columns},
        strides: strides({1, columns}, type),
        type: type
      }
      | do_list_rows(
          binary_part(
            rows,
            columns * element_size(type),
            (row_num - 1) * columns * element_size(type)
          ),
          row_num - 1,
          columns,
          type
        )
    ]
  end

  @doc """
  Load matrex from file.

  .csv and .mtx (binary) formats are supported.

  ## Example

      iex> Matrex.load("test/matrex.csv")
      #Matrex[5×4]
      ┌                                 ┐
      │     0.0  4.8e-4-0.00517-0.01552 │
      │-0.01616-0.01622 -0.0161-0.00574 │
      │  6.8e-4     0.0     0.0     0.0 │
      │     0.0     0.0     0.0     0.0 │
      │     0.0     0.0     0.0     0.0 │
      └                                 ┘
  """
  @spec load(binary) :: matrex
  def load(file_name) when is_binary(file_name) do
    cond do
      :filename.extension(file_name) == ".gz" ->
        File.read!(file_name)
        |> :zlib.gunzip()
        |> do_load(String.split(file_name, ".") |> Enum.at(-2) |> String.to_existing_atom())

      :filename.extension(file_name) == ".csv" ->
        do_load(File.read!(file_name), :csv)

      :filename.extension(file_name) == ".mtx" ->
        do_load(File.read!(file_name), :mtx)

      :filename.extension(file_name) == ".idx" ->
        do_load(File.read!(file_name), :idx)

      true ->
        raise "Unknown file format: #{file_name}"
    end
  end

  @spec load(binary, :idx | :csv | :mtx) :: matrex
  def load(file_name, format) when format in [:idx, :mtx, :csv],
    do: do_load(File.read!(file_name), format)

  defp do_load(data, :csv), do: new(data)
  # TODO: set matrix info
  defp do_load(data, :mtx), do: %Matrex{data: data, shape: {}, strides: {}, type: :float32}

  defp do_load(data, :idx),
    do: %Matrex{data: Matrex.IDX.load(data), shape: {}, strides: {}, type: :float32}

  @doc """
  Creates "magic" n*n matrix, where sums of all dimensions are equal.


  ## Example

      iex> Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
  """
  @spec magic(index, type) :: matrex
  def magic(n, type \\ :float32) when is_integer(n) and type in @types,
    do: Matrex.MagicSquare.new(n) |> new(type)

  @doc false
  # Shortcut to get functions list outside in Matrex.Operators module.
  def math_functions_list(), do: @math_functions

  @doc """
  Maximum element in a matrix. NIF.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.max(m)
      25.0

      iex> Matrex.reshape([1, 2, :inf, 4, 5, 6], 2, 3) |> max()
      :inf

  """
  @spec max(matrex) :: element
  def max(%Matrex{data: data, type: type}), do: call_nif(:max, type, [data])

  @doc """
  Returns maximum finite element of a matrex. NIF.

  Used on matrices which may contain infinite values.

  ## Example

      iex>Matrex.reshape([1, 2, :inf, 3, :nan, 5], 3, 2) |> Matrex.max_finite()
      5.0

  """
  @spec max_finite(matrex) :: number
  def max_finite(%Matrex{data: data, type: type}), do: call_nif(:max_finite, type, [data])

  @doc """

  Minimum element in a matrix. NIF.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.min(m)
      1.0

      iex> Matrex.reshape([1, 2, :neg_inf, 4, 5, 6], 2, 3) |> max()
      :neg_inf

  """
  @spec min(matrex) :: element
  def min(%Matrex{data: data, type: type}), do: call_nif(:min, type, [data])

  @doc """
  Returns minimum finite element of a matrex. NIF.

  Used on matrices which may contain infinite values.

  ## Example

      iex>Matrex.reshape([1, 2, :neg_inf, 3, 4, 5], 3, 2) |> Matrex.min_finite()
      1.0

  """
  @spec min_finite(matrex) :: number
  def min_finite(%Matrex{data: data, type: type}), do: call_nif(:min_finite, type, [data])

  types = [
    float64: {:float, 64},
    float32: {:float, 32},
    byte: {:integer, 8},
    int16: {:integer, 16},
    int32: {:integer, 32},
    int64: {:integer, 64}
  ]

  for {guard, type_and_size} <- types do
    @guard guard
    @type_and_size type_and_size

    @doc false
    def unquote(:"apply_on_matrix_#{@guard}")(<<>>, _, accumulator), do: accumulator

    def unquote(:"apply_on_matrix_#{@guard}")(
          <<value::type_and_size(), rest::binary>>,
          function,
          accumulator
        ) do
      new_value = function.(value)

      unquote(:"apply_on_matrix_#{@guard}")(
        rest,
        function,
        <<accumulator::binary, new_value::type_and_size()>>
      )
    end

    @doc false
    def unquote(:"apply_on_matrix_#{@guard}")(<<>>, _, _, _, accumulator), do: accumulator

    def unquote(:"apply_on_matrix_#{@guard}")(
          <<value::type_and_size(), rest::binary>>,
          function,
          pos,
          shape,
          accumulator
        ) do
      new_value = function.(value, pos)

      unquote(:"apply_on_matrix_#{@guard}")(
        rest,
        function,
        next_pos(pos, shape),
        shape,
        <<accumulator::binary, new_value::type_and_size()>>
      )
    end

    @doc false
    def unquote(:"apply_on_matrices_#{@guard}")(<<>>, <<>>, _, accumulator), do: accumulator

    def unquote(:"apply_on_matrices_#{@guard}")(
          <<first_value::type_and_size(), first_rest::binary>>,
          <<second_value::type_and_size(), second_rest::binary>>,
          function,
          accumulator
        )
        when is_function(function, 2) do
      new_value = function.(first_value, second_value)
      new_accumulator = <<accumulator::binary, new_value::type_and_size()>>

      unquote(:"apply_on_matrices_#{@guard}")(first_rest, second_rest, function, new_accumulator)
    end

    def at(%Matrex{data: data, strides: strides, type: @guard}, pos) when is_tuple(pos) do
      data
      |> binary_part(offset(strides, pos), element_size(@guard))
      |> unquote(:"binary_to_#{@guard}")()
    end

    def first(%Matrex{data: <<element::binary_size(), _::binary>>, type: @guard}),
      do: unquote(:"binary_to_#{@guard}")(element)

    @doc false
    def unquote(:"new_matrix_from_function_#{@guard}")(0, _, accumulator), do: accumulator

    def unquote(:"new_matrix_from_function_#{@guard}")(size, function, accumulator),
      do:
        unquote(:"new_matrix_from_function_#{@guard}")(
          size - 1,
          function,
          <<accumulator::binary, function.()::type_and_size()>>
        )

    @doc false
    def unquote(:"new_matrix_from_function_#{@guard}")(0, _, _, _, accumulator), do: accumulator

    def unquote(:"new_matrix_from_function_#{@guard}")(
          size,
          pos,
          shape,
          function,
          accumulator
        ) do
      new_accumulator = <<accumulator::binary, function.(pos)::type_and_size()>>

      unquote(:"new_matrix_from_function_#{@guard}")(
        size - 1,
        next_pos(pos, shape),
        shape,
        function,
        new_accumulator
      )
    end
  end

  @spec list_to_binary([element], type) :: binary
  defp list_to_binary(list, type) when is_list(list) and type in @types,
    do:
      Enum.reduce(list, <<>>, fn element, partial ->
        <<partial::binary, element_to_binary(element, type)::binary>>
      end)

  @doc """
  Elementwise multiplication of two matrices or matrix and a scalar. NIF.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.multiply(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     5.0     4.0     3.0 │
      │    12.0    20.0    36.0 │
      └                         ┘

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |> Matrex.multiply(2)
      #Matrex[2×3]
      ┌                         ┐
      │     2.0     4.0     6.0 │
      │     8.0    10.0    12.0 │
      └                         ┘

  """
  @spec multiply(matrex, matrex) :: matrex
  @spec multiply(matrex, number) :: matrex
  @spec multiply(number, matrex) :: matrex
  def multiply(%Matrex{data: first, shape: shape, strides: strides, type: type} = matrex, %Matrex{
        data: second,
        shape: shape,
        strides: strides,
        type: type
      }),
      do: %{matrex | data: call_nif(:multiply, type, [first, second])}

  def multiply(%Matrex{data: data, type: type} = matrex, scalar) when is_number(scalar),
    do: %{matrex | data: call_nif(:multiply_with_scalar, type, [data, scalar])}

  def multiply(scalar, %Matrex{data: data, type: type} = matrex) when is_number(scalar),
    do: %{matrex | data: call_nif(:multiply_with_scalar, type, [data, scalar])}

  @doc """
  Negates each element of the matrix. NIF.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |> Matrex.neg()
      #Matrex[2×3]
      ┌                         ┐
      │    -1.0    -2.0    -3.0 │
      │    -4.0    -5.0    -6.0 │
      └                         ┘

  """
  @spec neg(matrex) :: matrex
  def neg(%Matrex{data: matrix, type: type} = matrex),
    do: %{matrex | data: call_nif(:neg, type, [matrix])}

  @doc """
  Creates new matrix with values provided by the given function.

  If function accepts two arguments one-based row and column index of each element are passed to it.

  ## Examples

      iex> Matrex.new(3, 3, fn -> :rand.uniform() end)
      #Matrex[3×3]
      ┌                         ┐
      │ 0.45643 0.91533 0.25332 │
      │ 0.29095 0.21241  0.9776 │
      │ 0.42451 0.05422 0.92863 │
      └                         ┘

      iex> Matrex.new(3, 3, fn row, col -> row*col end)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     2.0     4.0     6.0 │
      │     3.0     6.0     9.0 │
      └                         ┘

  """
  @spec new(shape, (() -> element), type) :: matrex
  @spec new(shape, (tuple -> element), type) :: matrex
  def new(shape, function) when is_tuple(shape) and is_function(function),
    do: new(shape, function, :float32)

  def new(shape, function, type)
      when is_tuple(shape) and is_function(function, 0) and type in @types,
      do: %Matrex{
        data:
          call_typed(:new_matrix_from_function, type, [elements_count(shape), function, <<>>]),
        shape: shape,
        strides: strides(shape, type),
        type: type
      }

  def new(shape, function, type)
      when is_tuple(shape) and is_function(function, 1) and type in @types,
      do: %Matrex{
        data:
          call_typed(:new_matrix_from_function, type, [
            elements_count(shape),
            Tuple.duplicate(1, tuple_size(shape)),
            shape,
            function,
            <<>>
          ]),
        shape: shape,
        strides: strides(shape, type),
        type: type
      }

  @doc """
  Creates new matrix from list of lists or text representation (compatible with MathLab/Octave).

  List of lists can contain other matrices, which are concatenated in one.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.new([[Matrex.fill(2, 1.0), Matrex.fill(2, 3, 2.0)],
      ...> [Matrex.fill(1, 2, 3.0), Matrex.fill(1, 3, 4.0)]])
      #Matrex[5×5]
      ┌                                         ┐
      │     1.0     1.0     2.0     2.0     2.0 │
      │     1.0     1.0     2.0     2.0     2.0 │
      │     3.0     3.0     4.0     4.0     4.0 │
      └                                         ┘

      iex> Matrex.new("1;0;1;0;1")
      #Matrex[5×1]
      ┌         ┐
      │     1.0 │
      │     0.0 │
      │     1.0 │
      │     0.0 │
      │     1.0 │
      └         ┘

      iex> Matrex.new(\"\"\"
      ...>         1.00000   0.10000   0.60000   1.10000
      ...>         1.00000   0.20000   0.70000   1.20000
      ...>         1.00000       NaN   0.80000   1.30000
      ...>             Inf   0.40000   0.90000   1.40000
      ...>         1.00000   0.50000    NegInf   1.50000
      ...>       \"\"\")
      #Matrex[5×4]
      ┌                                 ┐
      │     1.0     0.1     0.6     1.1 │
      │     1.0     0.2     0.7     1.2 │
      │     1.0    NaN      0.8     1.3 │
      │     ∞       0.4     0.9     1.4 │
      │     1.0     0.5    -∞       1.5 │
      └                                 ┘

  """

  @spec new([[matrex]]) :: matrex
  def new(
        [
          [
            %Matrex{} | _
          ]
          | _
        ] = lol_of_ma
      ) do
    lol_of_ma
    |> Enum.map(&Matrex.concat/1)
    |> Enum.reduce(&Matrex.concat(&2, &1, :rows))
  end

  def new(source), do: new(source, :float32)

  @spec new([[element]] | binary, type) :: matrex
  def new([first_list | _] = lol_or_binary, type) when is_list(first_list) and type in @types do
    rows = length(lol_or_binary)
    columns = length(first_list)
    shape = {rows, columns}

    %Matrex{
      data:
        Enum.reduce(lol_or_binary, <<>>, fn list, accumulator ->
          accumulator <>
            Enum.reduce(list, <<>>, fn element, partial ->
              <<partial::binary, element_to_binary(element, type)::binary>>
            end)
        end),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }
  end

  def new(text, type) when is_binary(text) and type in @types do
    text
    |> String.split(["\n", ";"], trim: true)
    |> Enum.map(fn line ->
      line
      |> String.split(["\s", ","], trim: true)
      |> Enum.map(fn f -> parse_float(f) end)
    end)
    |> new()
  end

  @doc false
  @spec parse_float(binary) :: element | :nan | :inf | :neg_inf
  def parse_float("NaN"), do: :nan
  def parse_float("Inf"), do: :inf
  def parse_float("+Inf"), do: :inf
  def parse_float("-Inf"), do: :neg_inf
  def parse_float("NegInf"), do: :neg_inf

  def parse_float(string) do
    case Float.parse(string) do
      {value, _rem} -> value
      :error -> raise ArgumentError, message: "Unparseable matrix element value: #{string}"
    end
  end

  @doc """
  Bring all values of matrix into [0, 1] range. NIF.

  Where 0 corresponds to the minimum value of the matrix, and 1 — to the maxixmim.

  ## Example

      iex> m = Matrex.reshape(1..9, 3, 3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      └                         ┘
      iex> Matrex.normalize(m)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0   0.125    0.25 │
      │   0.375     0.5   0.625 │
      │    0.75   0.875     1.0 │
      └                         ┘
  """
  @spec normalize(matrex) :: matrex
  def normalize(%Matrex{data: data, type: type} = matrex) when type in @floats,
    do: %{matrex | data: call_nif(:normalize, type, [data])}

  @doc """
  Create matrix filled with ones.

  ## Example

      iex> Matrex.ones(2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(index, index) :: matrex
  def ones(rows, cols) when is_integer(rows) and is_integer(cols), do: fill(rows, cols, 1)

  @doc """
  Create matrex of ones of square dimensions or consuming output of `size/1` function.

  ## Examples

      iex> Matrex.ones(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘

      iex> m = Matrex.new("1 2 3; 4 5 6")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.ones(Matrex.size(m))
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(index) :: matrex
  @spec ones({index, index}) :: matrex
  def ones({rows, cols}), do: ones(rows, cols)

  def ones(size) when is_integer(size), do: fill(size, 1)

  @doc """
  Prints matrix to the console.

  Accepted options:
    * `:rows` — number of rows of matrix to show. Defaults to 21
    * `:columns` — number of columns of matrix to show. Defaults to maximum number of column,
    that fits into current terminal width.

    Returns the matrix itself, so can be used in pipes.

    ## Example

        iex> print(m, rows: 5, columns: 3)
        #Matrex[20×20]
        ┌                             ┐
        │     1.0   399.0  …     20.0 │
        │   380.0    22.0  …    361.0 │
        │   360.0    42.0  …    341.0 │
        │     ⋮       ⋮     …      ⋮  │
        │    40.0   362.0  …     21.0 │
        │   381.0    19.0  …    400.0 │
        └                             ┘

  """
  @spec print(matrex, Keyword.t()) :: matrex
  def print(%Matrex{} = matrex, opts \\ [rows: 21]) do
    {:ok, terminal_columns} = :io.columns()

    columns =
      case Keyword.get(opts, :columns) do
        nil -> terminal_columns
        cols -> cols * 8 + 10
      end

    matrex
    |> Matrex.Inspect.do_inspect(columns, Keyword.get(opts, :rows, 21))
    |> IO.puts()

    matrex
  end

  @doc """
  Create matrix of random floats in [0, 1] range. NIF.

  C language RNG is seeded on NIF libray load with `srandom(time(NULL) + clock())`.

  ## Example

      iex> Matrex.random(4,3)
      #Matrex[4×3]
      ┌                         ┐
      │ 0.32994 0.28736 0.88012 │
      │ 0.51782 0.68608 0.29976 │
      │ 0.52953  0.9071 0.26743 │
      │ 0.82189 0.59311  0.8451 │
      └                         ┘

  """
  @spec random(index, index) :: matrex
  def random(rows, columns) when is_integer(rows) and is_integer(columns),
    do: random({rows, columns}, :float32)

  @spec random(shape, type) :: matrex
  def random(shape, type) when is_tuple(shape) and type in @types,
    do: %Matrex{
      data: call_nif(:random, type, [elements_count(shape)]),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }

  @doc """
  Create square matrix of random floats.

  See `random/2` for details.

  ## Example

      iex> Matrex.random(3)
      #Matrex[3×3]
      ┌                         ┐
      │ 0.66438 0.31026 0.98602 │
      │ 0.82127 0.04701 0.13278 │
      │ 0.96935 0.70772 0.98738 │
      └                         ┘
  """
  @spec random(index) :: matrex
  def random(size) when is_integer(size), do: random({size, size}, :float32)
  def random(shape) when is_tuple(shape), do: random(shape, :float32)

  @doc """
  Resize matrix by scaling its dimenson with `scale`. NIF.

  ## Examples

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex(3)> Matrex.resize(m, 2)
      #Matrex[6×6]
      ┌                                                 ┐
      │     8.0     8.0     1.0     1.0     6.0     6.0 │
      │     8.0     8.0     1.0     1.0     6.0     6.0 │
      │     3.0     3.0     5.0     5.0     7.0     7.0 │
      │     3.0     3.0     5.0     5.0     7.0     7.0 │
      │     4.0     4.0     9.0     9.0     2.0     2.0 │
      │     4.0     4.0     9.0     9.0     2.0     2.0 │
      └                                                 ┘

      iex(4)> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex(5)> Matrex.resize(m, 0.5)
      #Matrex[3×3]
      ┌                         ┐
      │    16.0    23.0     7.0 │
      │    22.0     4.0    13.0 │
      │     9.0    11.0    25.0 │
      └                         ┘
  """
  @spec resize(matrex, number, :nearest | :bilinear) :: matrex
  def resize(matrex, scale, method \\ :nearest)

  def resize(%Matrex{} = matrex, 1, _), do: matrex

  def resize(%Matrex{data: data, shape: {rows, cols}, type: type} = matrex, scale, :nearest)
      when is_number(scale) and scale > 0,
      do: %{matrex | data: call_nif(:resize, type, [data, rows, cols, scale])}

  @doc """
  Reshapes list of values into a matrix of given size or changes the shape of existing matrix.

  Takes a list or anything, that implements `Enumerable.to_list/1`.

  Can take a list of matrices and concatenate them into one big matrix.

  Raises `ArgumentError` if list size and given shape do not match.

  ## Example

      iex> [1, 2, 3, 4, 5, 6] |> Matrex.reshape(2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.reshape([Matrex.zeros(2), Matrex.ones(2),
      ...> Matrex.fill(3, 2, 2.0), Matrex.fill(3, 2, 3.0)], 2, 2)
      #Matrex[5×4]
      ┌                                 ┐
      │     0.0     0.0     1.0     1.0 │
      │     0.0     0.0     1.0     1.0 │
      │     2.0     2.0     3.0     3.0 │
      │     2.0     2.0     3.0     3.0 │
      │     2.0     2.0     3.0     3.0 │
      └                                 ┘

      iex> Matrex.reshape(1..6, 2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘

      iex> Matrex.new("1 2 3; 4 5 6") |> Matrex.reshape(3, 2)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘

  """
  def reshape(source, shape, type \\ :float32)

  def reshape([], _, _), do: raise(ArgumentError, message: "Empty list cannot be reshaped.")

  @spec reshape([matrex], shape) :: matrex
  def reshape([%Matrex{} | _] = enum, {_rows, columns}, _type) do
    enum
    |> Enum.chunk(columns)
    |> new()
  end

  @spec reshape([element], shape, type) :: matrex
  def reshape([_ | _] = list, shape, type) when is_tuple(shape) and type in @types do
    data = list_to_binary(list, type)

    if byte_size(data) / element_size(type) != elements_count(shape),
      do:
        raise(
          ArgumentError,
          message:
            "Cannot reshape: #{byte_size(data)} bytes do not fit into #{inspect(shape)} of type #{
              type
            }."
        ),
      else: %Matrex{
        data: data,
        shape: shape,
        strides: strides(shape, type),
        type: type
      }
  end

  def reshape(%Matrex{shape: shape} = matrex, shape, _type),
    # No need to reshape.
    do: matrex

  def reshape(
        %Matrex{shape: shape} = matrex,
        new_shape,
        type
      )
      when is_tuple(new_shape),
      do:
        if(
          elements_count(shape) != elements_count(new_shape),
          do:
            raise(
              ArgumentError,
              message:
                "Cannot reshape: #{inspect(shape)} does not fit into #{inspect(new_shape)}."
            ),
          else: %{matrex | shape: new_shape, strides: strides(new_shape, type)}
        )

  def reshape(a..b, shape, type) when is_tuple(shape) and type in @types,
    do:
      if(
        b - a + 1 != elements_count(shape),
        do:
          raise(
            ArgumentError,
            message: "range #{a}..#{b} cannot be reshaped into #{inspect(shape)} matrix."
          ),
        else: %Matrex{
          data: call_nif(:from_range, type, [a, b]),
          shape: shape,
          strides: strides(shape, type),
          type: type
        }
      )

  def reshape(input, shape, type) when is_tuple(shape) and type in @types,
    do: input |> Enum.to_list() |> reshape(shape, type)

  @doc """
  Return matrix row as list by one-based index.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.row_to_list(m, 3)
      [3.0, 10.0, 12.0, 19.0, 21.0]
  """
  @spec row_to_list(matrex, index) :: [element]
  def row_to_list(%Matrex{data: matrix, type: type}, row) when is_integer(row) and row > 0,
    do: call_nif(:row_to_list, type, [matrix, row - 1])

  @doc """
  Get row of matrix as matrix (vector) in matrex form. One-based.

  You can use shorter `matrex[n]` syntax for the same result.

  ## Example

      iex> m = Matrex.magic(5)
      #Matrex[5×5]
      ┌                                         ┐
      │    16.0    23.0     5.0     7.0    14.0 │
      │    22.0     4.0     6.0    13.0    20.0 │
      │     3.0    10.0    12.0    19.0    21.0 │
      │     9.0    11.0    18.0    25.0     2.0 │
      │    15.0    17.0    24.0     1.0     8.0 │
      └                                         ┘
      iex> Matrex.row(m, 4)
      #Matrex[1×5]
      ┌                                         ┐
      │     9.0    11.0    18.0    25.0     2.0 │
      └                                         ┘
      iex> m[4]
      #Matrex[1×5]
      ┌                                         ┐
      │     9.0    11.0    18.0    25.0     2.0 │
      └                                         ┘
  """
  @spec row(matrex, index) :: matrex
  def row(%Matrex{data: data, shape: {rows, columns}, type: type}, row)
      when is_integer(row) and row > 0 and row <= rows,
      do: %Matrex{
        data:
          binary_part(
            data,
            (row - 1) * columns * element_size(type),
            columns * element_size(type)
          ),
        shape: {1, columns},
        strides: strides({1, columns}, type),
        type: type
      }

  @doc """
  Saves matrex into file.

  Binary (.mtx) and CSV formats are supported currently.

  Format is defined by the extension of the filename.

  ## Example

      iex> Matrex.random(5) |> Matrex.save("r.mtx")
      :ok
  """
  @spec save(matrex, binary) :: :ok | :error
  def save(
        %Matrex{
          data: data
        },
        file_name
      )
      when is_binary(file_name) do
    cond do
      :filename.extension(file_name) == ".mtx" ->
        File.write!(file_name, data)

      :filename.extension(file_name) == ".csv" ->
        csv =
          data
          |> NIFs.to_list_of_lists()
          |> Enum.reduce("", fn row_list, acc ->
            acc <>
              Enum.reduce(row_list, "", fn elem, line ->
                line <> element_to_string(elem) <> ","
              end) <> "\n"
          end)

        File.write!(file_name, csv)

      true ->
        raise "Unknown file format: #{file_name}"
    end
  end

  @doc false
  @spec element_to_string(element) :: binary
  # Save zero values without fraction part to save space
  def element_to_string(0.0), do: "0"
  def element_to_string(val) when is_float(val), do: Float.to_string(val)
  def element_to_string(val) when is_integer(val), do: Integer.to_string(val)
  def element_to_string(:nan), do: "NaN"
  def element_to_string(:inf), do: "Inf"
  def element_to_string(:neg_inf), do: "-Inf"

  @doc """
  Transfer one-element matrix to a scalar value.

  Differently from `first/1` will not match and throw an error,
  if matrix contains more than one element.

  ## Example

      iex> Matrex.new([[1.234]]) |> Matrex.scalar()
      1.234

      iex> Matrex.new([[0]]) |> Matrex.divide(0) |> Matrex.scalar()
      :nan

      iex> Matrex.new([[1.234, 5.678]]) |> Matrex.scalar()
      ** (FunctionClauseError) no function clause matching in Matrex.scalar/1
  """
  @spec scalar(matrex) :: element
  def scalar(%Matrex{
        data: <<elem::binary>>,
        shape: shape,
        type: type
      }),
      do:
        if(
          byte_size(elem) != element_size(type) or elements_count(shape) != 1,
          do:
            raise(
              ArgumentError,
              message: "Matrix contains more than one scalar."
            ),
          else: binary_to_element(elem, type)
        )

  @doc """
  Set element of matrix at the specified position (one-based) to new value.

  ## Example

      iex> m = Matrex.ones(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
      iex> m = Matrex.set(m, 2, 2, 0)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
      iex> m = Matrex.set(m, 3, 2, :neg_inf)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     -∞      1.0 │
      └                         ┘
  """
  @spec set(matrex, position, element) :: matrex
  def set(%Matrex{data: data, shape: shape, strides: strides, type: type} = matrex, pos, value)
      when is_number(value) or (value in [:nan, :inf, :neg_inf] and is_tuple(pos)),
      do: %{
        matrex
        | data: call_nif(:set, type, [data, offset(strides, pos), element_to_binary(value, type)])
      }

  def set(%Matrex{shape: {_rows, _cols}} = matrex, row, column, value),
    do: set(matrex, {row, column}, value)

  @doc """
  Set column of a matrix to the values from the given 1-column matrix. NIF.

  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[3×2]
      ┌                    ┐
      │     1.0     2.0    │
      │     3.0     4.0    │
      │     5.0     6.0    │
      └                    ┘

      iex> Matrex.set_column(m, 2, Matrex.new("7; 8; 9"))
      #Matrex[3×2]
      ┌                    ┐
      │     1.0     7.0    │
      │     3.0     8.0    │
      │     5.0     9.0    │
      └                    ┘
  """
  @spec set_column(matrex, index, matrex) :: matrex
  def set_column(
        %Matrex{data: data, shape: {rows, columns}, type: type} = matrex,
        column,
        %Matrex{
          data: column_data,
          shape: {rows, 1},
          type: type
        }
      )
      when column in 1..columns,
      do: %{matrex | data: call_nif(:set_column, type, [data, column - 1, column_data])}

  @doc """
  Return size of matrix as `{rows, cols}`

  ## Example

      iex> m = Matrex.random(2,3)
      #Matrex[2×3]
      ┌                         ┐
      │ 0.69745 0.23668 0.36376 │
      │ 0.63423 0.29651 0.22844 │
      └                         ┘
      iex> Matrex.size(m)
      {2, 3}
  """
  @spec size(matrex) :: shape
  def size(%Matrex{shape: shape}), do: shape

  @doc """
  Produces element-wise squared matrix. NIF through `multiply/4`.


  ## Example

      iex> m = Matrex.new("1 2 3; 4 5 6")
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.square(m)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     4.0     9.0 │
      │    16.0    25.0    36.0 │
      └                         ┘

  """
  @spec square(matrex) :: matrex
  def square(%Matrex{data: data, type: type} = matrex),
    do: %{matrex | data: call_nif(:multiply, type, [data, data])}

  @doc """
  Returns submatrix for a given matrix. NIF.

  Rows and columns ranges are inclusive and one-based.

  ## Example

      iex> m = Matrex.new("1 2 3; 4 5 6; 7 8 9")
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      │     7.0     8.0     9.0 │
      └                         ┘
      iex> Matrex.submatrix(m, 2..3, 2..3)
      #Matrex[2×2]
      ┌                ┐
      │    5.0     6.0 │
      │    8.0     9.0 │
      └                ┘
  """
  @spec submatrix(matrex, Range.t(), Range.t()) :: matrex
  def submatrix(
        %Matrex{data: data, shape: {rows, cols}, type: type},
        row_from..row_to,
        col_from..col_to
      )
      when row_from in 1..rows and row_to in row_from..rows and col_from in 1..cols and
             col_to in col_from..cols,
      do: %Matrex{
        data:
          call_nif(:submatrix, type, [data, row_from - 1, row_to - 1, col_from - 1, col_to - 1]),
        shape: {row_to - row_from + 1, col_to - col_from + 1},
        strides: strides({row_to - row_from + 1, col_to - col_from + 1}, type),
        type: type
      }

  def submatrix(%Matrex{} = matrex, rows, cols) do
    raise(
      RuntimeError,
      "Submatrix position out of range or malformed: position is " <>
        "(#{Kernel.inspect(rows)}, #{Kernel.inspect(cols)}), source size is " <>
        "(#{Kernel.inspect(1..matrex[:rows])}, #{Kernel.inspect(1..matrex[:columns])})"
    )
  end

  @doc """
  Subtracts two matrices or matrix from scalar element-wise. NIF.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Examples

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.subtract(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │    -4.0     0.0     2.0 │
      │     1.0     1.0     0.0 │
      └                         ┘

      iex> Matrex.subtract(1, Matrex.new([[1, 2, 3], [4, 5, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     0.0     -1.0   -2.0 │
      │    -3.0    -4.0    -5.0 │
      └                         ┘
  """
  @spec subtract(matrex | number, matrex | number) :: matrex
  def subtract(%Matrex{data: first, shape: shape, strides: strides, type: type} = matrex, %Matrex{
        data: second,
        shape: shape,
        strides: strides,
        type: type
      }),
      do: %{matrex | data: call_nif(:subtract, type, [first, second])}

  def subtract(scalar, %Matrex{data: data, type: type} = matrex) when is_number(scalar),
    do: %{matrex | data: call_nif(:subtract_from_scalar, type, [scalar, data])}

  def subtract(%Matrex{data: data, type: type} = matrex, scalar) when is_number(scalar),
    do: %{matrex | data: call_nif(:add_scalar, type, [data, -scalar])}

  @doc """
  Subtracts the second matrix or scalar from the first. Inlined.

  Raises `ErlangError` if matrices' sizes do not match.

  ## Example

      iex> Matrex.new([[1, 2, 3], [4, 5, 6]]) |>
      ...> Matrex.subtract_inverse(Matrex.new([[5, 2, 1], [3, 4, 6]]))
      #Matrex[2×3]
      ┌                         ┐
      │     4.0     0.0    -2.0 │
      │    -1.0    -1.0     0.0 │
      └                         ┘

      iex> Matrex.eye(3) |> Matrex.subtract_inverse(1)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0     1.0     1.0 │
      │     1.0     0.0     1.0 │
      │     1.0     1.0     0.0 │
      └                         ┘

  """
  @spec subtract_inverse(matrex | number, matrex | number) :: matrex
  def subtract_inverse(%Matrex{} = first, %Matrex{} = second), do: subtract(second, first)

  def subtract_inverse(%Matrex{} = first, scalar) when is_number(scalar),
    do: subtract(scalar, first)

  def subtract_inverse(scalar, %Matrex{} = second) when is_number(scalar),
    do: subtract(second, scalar)

  @doc """
  Sums all elements. NIF.

  Can return special float values as atoms.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.sum(m)
      45.0

      iex> m = Matrex.new("1 Inf; 2 3")
      #Matrex[2×2]
      ┌                 ┐
      │     1.0     ∞   │
      │     2.0     3.0 │
      └                 ┘
      iex> sum(m)
      :inf
  """
  @spec sum(matrex) :: element
  def sum(%Matrex{data: data, type: type}), do: call_nif(:sum, type, [data])

  @doc """
  Converts to flat list. NIF.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_list(m)
      [8.0, 1.0, 6.0, 3.0, 5.0, 7.0, 4.0, 9.0, 2.0]
  """
  @spec to_list(matrex) :: list(element)
  def to_list(%Matrex{data: data, type: type}), do: call_nif(:to_list, type, [data])

  @doc """
  Converts to list of lists. NIF.

  ## Examples

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_list_of_lists(m)
      [[8.0, 1.0, 6.0], [3.0, 5.0, 7.0], [4.0, 9.0, 2.0]]

      iex> r = Matrex.divide(Matrex.eye(3), Matrex.zeros(3))
      #Matrex[3×3]
      ┌                         ┐
      │     ∞      NaN     NaN  │
      │    NaN      ∞      NaN  │
      │    NaN     NaN      ∞   │
      └                         ┘
      iex> Matrex.to_list_of_lists(r)
      [[:inf, :nan, :nan], [:nan, :inf, :nan], [:nan, :nan, :inf]]

  """
  @spec to_list_of_lists(matrex) :: list(list(element))
  def to_list_of_lists(%Matrex{data: data, shape: {rows, cols}, type: type}),
    do: call_nif(:to_list_of_lists, type, [data, rows, cols])

  @doc """
  Convert any matrix m×n to a column matrix (m*n)×1.

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_column(m)
      #Matrex[1×9]
      ┌                                                                         ┐
      │     8.0     1.0     6.0     3.0     5.0     7.0     4.0     9.0     2.0 │
      └                                                                         ┘

  """
  @spec to_column(matrex) :: matrex
  def to_column(%Matrex{shape: {_rows, 1}} = m), do: m
  def to_column(%Matrex{shape: {rows, columns}} = m), do: reshape(m, {rows * columns, 1})

  @doc """
  Convert any matrix m×n to a row matrix 1×(m*n).

  ## Example

      iex> m = Matrex.magic(3)
      #Matrex[3×3]
      ┌                         ┐
      │     8.0     1.0     6.0 │
      │     3.0     5.0     7.0 │
      │     4.0     9.0     2.0 │
      └                         ┘
      iex> Matrex.to_row(m)
      #Matrex[1×9]
      ┌                                                                         ┐
      │     8.0     1.0     6.0     3.0     5.0     7.0     4.0     9.0     2.0 │
      └                                                                         ┘

  """
  @spec to_row(matrex) :: matrex
  def to_row(%Matrex{shape: {1, _columns}} = m), do: m
  def to_row(%Matrex{shape: {rows, columns}} = m), do: reshape(m, {1, rows * columns})

  @doc """
  Transposes a matrix. NIF.

  ## Example

      iex> m = Matrex.new([[1,2,3],[4,5,6]])
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      │     4.0     5.0     6.0 │
      └                         ┘
      iex> Matrex.transpose(m)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     4.0 │
      │     2.0     5.0 │
      │     3.0     6.0 │
      └                 ┘
  """
  @spec transpose(matrex) :: matrex
  # Vectors are transposed by simply reshaping
  def transpose(%Matrex{shape: {1, columns}} = m), do: reshape(m, columns, 1)
  def transpose(%Matrex{shape: {rows, 1}} = m), do: reshape(m, 1, rows)

  def transpose(%Matrex{data: data, shape: {rows, cols}, type: type}),
    do: %Matrex{
      data: call_nif(:transpose, type, [data, rows, cols]),
      shape: {cols, rows},
      strides: strides({cols, rows}, type),
      type: type
    }

  @doc """
  Updates the element at the given position in matrix with function.

  Function is invoked with the current element value


  ## Example

      iex> m = Matrex.reshape(1..6, 3, 2)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0     4.0 │
      │     5.0     6.0 │
      └                 ┘
      iex> Matrex.update(m, 2, 2, fn x -> x * x end)
      #Matrex[3×2]
      ┌                 ┐
      │     1.0     2.0 │
      │     3.0    16.0 │
      │     5.0     6.0 │
      └                 ┘

  """
  @spec update(matrex, position, (element -> element)) :: matrex
  def update(%Matrex{data: data, shape: shape, strides: strides, type: type} = matrex, pos, fun)
      when is_function(fun, 1) do
    if(not inside_shape(pos, shape)) do
      raise(
        ArgumentError,
        message: "Position (#{inspect(pos)}) is out of matrex of shape #{inspect(shape)}"
      )
    else
      new_value =
        data
        |> binary_part(offset(strides, pos), element_size(type))
        |> binary_to_element(type)
        |> fun.()
        |> element_to_binary(type)

      %{matrex | data: call_nif(:set, type, [data, offset(strides, pos), new_value])}
    end
  end

  @doc """
  Create matrix of zeros of the specified size. NIF, using `memset()`.

  Faster, than `fill(rows, cols, 0)`.

  ## Example

      iex> Matrex.zeros(4,3)
      #Matrex[4×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(shape, type) :: matrex
  def zeros(shape, type \\ :float32)

  def zeros(shape, type) when is_tuple(shape) and type in @types,
    do: %Matrex{
      data: NIFs.zeros(elements_count(shape) * element_size(type)),
      shape: shape,
      strides: strides(shape, type),
      type: type
    }

  @doc """
  Create square matrix of size `size` rows × `size` columns, filled with zeros. Inlined.

  ## Example

      iex> Matrex.zeros(3)
      #Matrex[3×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(index, type) :: matrex
  def zeros(size, type) when is_integer(size), do: zeros({size, size}, type)
end
