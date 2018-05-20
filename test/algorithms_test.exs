defmodule AlgorithmsTest do
  ExUnit.start()
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

    expected_j = 2.5348193645477295

    expected_grad =
      Matrex.new(
        "0.1465613692998886; -0.5485584139823914; 0.7247222661972046; 1.3980028629302979"
      )

    # On Travis CI the first element is computed a bit differently.
    expected_grad2 =
      Matrex.new(
        "0.14656135439872742; -0.5485584139823914; 0.7247222661972046; 1.3980028629302979"
      )

    {j, grad} = Algorithms.lr_cost_fun(theta_t, {x_t, y_t, lambda_t})

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j

    {j, grad} = Algorithms.lr_cost_fun_ops(theta_t, {x_t, y_t, lambda_t})

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j
  end

  @tag timeout: 120_000
  test "#fmincg does linear regression" do
    x = Matrex.load("test/data/X.mtx.gz") |> Matrex.normalize()
    y = Matrex.load("test/data/Y.mtx")

    x = Matrex.concat(Matrex.ones(x[:rows], 1), x)
    theta = Matrex.zeros(x[:cols], 1)

    lambda = 0.1
    iterations = 100

    IO.write(IO.ANSI.clear())

    solutions =
      1..10
      |> Task.async_stream(
        fn digit ->
          y3 = Matrex.apply(y, fn val -> if(val == digit, do: 1.0, else: 0.0) end)

          {sX, fX, _i} =
            Algorithms.fmincg(&Algorithms.lr_cost_fun/2, theta, {x, y3, lambda}, iterations)

          {digit, List.last(fX), sX}
        end,
        max_concurrency: 2,
        timeout: 100_000
      )
      |> Enum.map(fn {:ok, {_d, _l, theta}} -> Matrex.to_list(theta) end)
      |> Matrex.new()

    # Visualize solutions
    # solutions
    # |> Matrex.to_list_of_lists()
    # |> Enum.each(&(Matrex.reshape(tl(&1), 28, 28) |> Matrex.heatmap()))

    # x_test = Matrex.load("test/data/Xtest.mtx.gz") |> Matrex.normalize()
    # x_test = Matrex.concat(Matrex.ones(x_test[:rows], 1), x_test)

    predictions =
      x
      |> Matrex.dot_nt(solutions)
      |> Matrex.apply(:sigmoid)

    # |> IO.inspect(label: "Predictions")

    # y_test = Matrex.load("test/data/Ytest.mtx")

    accuracy =
      1..predictions[:rows]
      |> Enum.reduce(0, fn row, acc ->
        if y[row] == predictions[row][:argmax], do: acc + 1, else: acc
      end)
      |> Kernel./(predictions[:rows])
      |> Kernel.*(100)
      |> IO.inspect(label: "\rTraining set accuracy")

    assert accuracy >= 95
  end
end
