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

  @input_layer_size 20 * 20
  @hidden_layer_size 25
  @num_labels 10

  test "#nn_cost_function computes neural network cost with and w/0 regularization" do
    x = Matrex.load("test/data/X.mtx.gz")
    y = Matrex.load("test/data/Y.mtx")
    theta1 = Matrex.load("../../Octave/ex4/theta1.csv") |> Matrex.to_row()
    theta2 = Matrex.load("../../Octave/ex4/theta2.csv") |> Matrex.to_row()

    theta = Matrex.concat(theta1, theta2) |> Matrex.transpose()

    lambda = 0

    {j, grads} =
      Matrex.Algorithms.nn_cost_fun(
        theta,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda}
      )

    assert j == 0.2876291611777169
    lambda = 1

    {j, grads} =
      Matrex.Algorithms.nn_cost_fun(
        theta,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda}
      )

    assert j == 0.3837698553161823
  end

  @tag timeout: 600_000
  test "#fmincg optimizes neural network" do
    initial_theta1 = random_weights(@input_layer_size, @hidden_layer_size) |> Matrex.to_row()
    initial_theta2 = random_weights(@hidden_layer_size, @num_labels) |> Matrex.to_row()
    initial_nn_params = Matrex.concat(initial_theta1, initial_theta2) |> Matrex.transpose()

    x = Matrex.load("test/data/X.mtx.gz")
    y = Matrex.load("test/data/Y.mtx")

    lambda = 0.5
    iterations = 10

    {sX, _fX, _i} =
      Algorithms.fmincg(
        &Algorithms.nn_cost_fun/2,
        initial_nn_params,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda},
        iterations
      )

    # Unpack thetas from the found solution
    theta1 =
      sX[1..(@hidden_layer_size * (@input_layer_size + 1))]
      |> Matrex.reshape(@hidden_layer_size, @input_layer_size + 1)

    theta2 =
      sX[(@hidden_layer_size * (@input_layer_size + 1) + 1)..sX[:rows]]
      |> Matrex.reshape(@num_labels, @hidden_layer_size + 1)

    theta1h = Matrex.submatrix(theta1, 1..theta1[:rows], 2..theta1[:cols])

    theta1h
    |> Matrex.Algorithms.visual_net({5, 5}, {20, 20})
    |> Matrex.heatmap()

    predictions = Matrex.Algorithms.nn_predict(theta1, theta2, x)

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
  end

  defp random_weights(l_in, l_out) do
    epsilon_init = 0.12

    Matrex.random(l_out, 1 + l_in)
    |> Matrex.multiply(2 * epsilon_init)
    |> Matrex.substract(epsilon_init)
  end
end
