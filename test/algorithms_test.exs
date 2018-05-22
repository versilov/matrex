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

  @tag skip: true
  @tag timeout: 120_000
  test "#fmincg does linear regression" do
    x = Matrex.load("test/data/X.mtx.gz") |> Matrex.normalize()
    y = Matrex.load("test/data/Y.mtx")

    x = Matrex.concat(Matrex.ones(x[:rows], 1), x)
    theta = Matrex.zeros(x[:cols], 1)

    lambda = 0.01
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
        max_concurrency: 1,
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
        if y[row] == predictions[row][:argmax] do
          acc + 1
        else
          # Show wrongful predictions
          # x[row][2..785] |> Matrex.reshape(28, 28) |> Matrex.heatmap()
          # IO.puts("#{y[row]} != #{predictions[row][:argmax]}")
          acc
        end
      end)
      |> Kernel./(predictions[:rows])
      |> Kernel.*(100)
      |> IO.inspect(label: "\rTraining set accuracy")

    assert accuracy >= 95
  end

  @sample_side_size 20
  @input_layer_size @sample_side_size * @sample_side_size
  @hidden_layer_size 25
  @num_labels 10

  test "#nn_cost_function computes neural network cost with and w/0 regularization" do
    x = Matrex.load("test/data/X.mtx.gz")
    y = Matrex.load("test/data/Y.mtx")
    theta1 = Matrex.load("test/data/nn_theta1.mtx") |> Matrex.to_row()
    theta2 = Matrex.load("test/data/nn_theta2.mtx") |> Matrex.to_row()

    theta = Matrex.concat(theta1, theta2) |> Matrex.transpose()

    lambda = 0

    {j, _grads} =
      Matrex.Algorithms.nn_cost_fun(
        theta,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda}
      )

    assert round_enough(j) == round_enough(0.287629150390625)
    lambda = 1

    {j, _grads} =
      Matrex.Algorithms.nn_cost_fun(
        theta,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda}
      )

    assert round_enough(j) == round_enough(0.3837698553161823)
  end

  @tag timeout: 600_000
  @tag skip: true
  test "#fmincg optimizes neural network" do
  end

  defp round_enough(num), do: Float.round(num, 5)

  # Split data into training and testing set, permute it randomly
  @spec split_data(Matrex.t(), Matrex.t()) :: {Matrex.t(), Matrex.t(), Matrex.t(), Matrex.t()}
  defp split_data(x, y) do
    n = x[:rows]
    n_train = trunc(0.8 * n)
    n_test = n - n_train
    n_rows = Enum.take_random(1..n, n)

    {x_train, y_train, rows} =
      Enum.reduce(2..n_train, {x[hd(n_rows)], Matrex.row(y, hd(n_rows)), tl(n_rows)}, fn _i,
                                                                                         {x_train,
                                                                                          y_train,
                                                                                          rows} ->
        {Matrex.concat(x_train, x[hd(rows)], :rows),
         Matrex.concat(y_train, Matrex.row(y, hd(rows)), :rows), tl(rows)}
      end)

    {x_test, y_test, _rows} =
      Enum.reduce(1..(n_test - 2), {x[hd(rows)], Matrex.row(y, hd(rows)), tl(rows)}, fn _i,
                                                                                        {x_test,
                                                                                         y_test,
                                                                                         rows} ->
        {Matrex.concat(x_test, x[hd(rows)], :rows),
         Matrex.concat(y_test, Matrex.row(y, hd(rows)), :rows), tl(rows)}
      end)

    {x_train, y_train, x_test, y_test}
  end
end
