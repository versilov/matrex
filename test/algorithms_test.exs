defmodule AlgorithmsTest do
  use ExUnit.Case, async: true

  alias Matrex.Algorithms

  test "#lr_cost_fun computes cost" do
    theta_t = Matrex.new([[-2], [-1], [1], [2]])

    x_t =
      Matrex.new("""
        1.00000   0.10000   0.60000   1.10000
        1.00000   0.20000   0.70000   1.20000
        1.00000   0.30000   0.80000   1.30000
        1.00000   0.40000   0.90000   1.40000
        1.00000   0.50000   1.00000   1.50000
      """)

    y_t = Matrex.new("1;0;1;0;1")
    lambda_t = 3
    {j, grad} = lr_cost_fun(theta_t, {x_t, y_t, lambda_t})

    assert j == 2.5348193645477295

    assert grad ==
             Matrex.new(
               "0.1465613692998886; -0.5485584139823914; 0.7247222661972046; 1.3980029821395874"
             )
  end

  test "#fmincg" do
    x = Matrex.load("test/X.mtx")
    y = Matrex.load("test/y.mtx")
    theta = Matrex.zeros(x[:cols], 1)

    lambda = 0.01
    iterations = 100

    1..10
    |> Task.async_stream(
      fn digit ->
        y3 = Matrex.apply(y, fn val -> if(val == digit, do: 1.0, else: 0.0) end)

        {_sX, fX, _i} = Algorithms.fmincg(&lr_cost_fun/2, theta, {x, y3, lambda}, iterations)
        {digit, List.last(fX)}
      end,
      max_concurrency: 4
    )
    |> Enum.to_list()
    |> IO.inspect()

    # for digit <- 1..10 do
    #   y3 = Matrex.apply(y, fn val -> if(val == digit, do: 1.0, else: 0.0) end)
    #
    #   {_sX, fX, _i} = Algorithms.fmincg(&lr_cost_fun/2, theta, {x, y3, lambda}, iterations)
    #   IO.inspect(List.last(fX), label: "#{digit}")
    # end
  end

  defp lr_cost_fun(%Matrex{} = theta, {%Matrex{} = x, %Matrex{} = y, lambda})
       when is_number(lambda) do
    m = y[:rows]

    h = Matrex.dot(x, theta) |> Matrex.apply(:sigmoid)
    l = Matrex.ones(theta[:rows], theta[:cols]) |> Matrex.set(1, 1, 0)

    normalization =
      Matrex.dot_tn(l, Matrex.multiply(theta, theta))
      |> Matrex.first()
      |> Kernel.*(lambda / (2 * m))

    j =
      y
      |> Matrex.multiply(-1)
      |> Matrex.dot_tn(Matrex.apply(h, :log))
      |> Matrex.substract(
        Matrex.dot_tn(
          Matrex.substract(1, y),
          Matrex.apply(Matrex.substract(1, h), :log)
        )
      )
      |> Matrex.first()
      |> (fn
            NaN -> NaN
            x -> x / m + normalization
          end).()

    grad =
      x
      |> Matrex.dot_tn(Matrex.substract(h, y))
      |> Matrex.add(Matrex.multiply(Matrex.multiply(theta, l), lambda))
      |> Matrex.multiply(1 / m)

    {j, grad}
  end
end
