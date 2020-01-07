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

    {j, grad} = Algorithms.lr_cost_fun(theta_t, {x_t, y_t, lambda_t, 0})

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j

    {j, grad} = Algorithms.lr_cost_fun_ops(theta_t, {x_t, y_t, lambda_t})

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j
  end

  @tag skip: false
  @tag timeout: 120_000
  test "#fmincg does linear regression" do
    accuracy = Algorithms.run_lr(100, 5)
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

    assert Float.round(j, 7) == Float.round(0.287629150390625, 7)
    lambda = 1

    {j, _grads} =
      Matrex.Algorithms.nn_cost_fun(
        theta,
        {@input_layer_size, @hidden_layer_size, @num_labels, x, y, lambda}
      )

    assert Float.round(j, 6) == Float.round(0.38376984558105465, 6)
  end

  @tag timeout: 600_000
  @tag skip: true
  test "#fmincg optimizes neural network" do
  end

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


  test "#linear_cost_fun computes cost" do
    m = Matrex.load("test/rand_array.mtx")
    y_t = m |> Matrex.submatrix(1..41, 2..2)

    # for linear func, must add `ones` for the offset constant
    x = m |> Matrex.submatrix(1..41, 1..1)
    x_t = Matrex.concat(Matrex.ones(Matrex.size(x)), x)

    lambda_t = 0.01
    theta_t = Matrex.zeros(2, 1)

    expected_j = 5238.50381097561

    expected_grad =
      Matrex.new(
        " -0.91246 ; -2.41489 "
      )

    {j, grad} = Algorithms.linear_cost_fun(theta_t, {x_t, y_t, lambda_t})

    assert grad |> Matrex.subtract(expected_grad) |> Matrex.sum() < 5.0e-6
    assert j == expected_j

  end

  test "#fit_poly " do
    m = Matrex.load("test/rand_array.mtx")
    y = m |> Matrex.submatrix(1..41, 2..2)
    x = m |> Matrex.submatrix(1..41, 1..1)

    fit = Algorithms.fit_poly(x, y, 2)

    expected_fit = %{
      coefs: [
        {0, 37.48050308227539},
        {1, 6.260676383972168},
        {2, 6.991103172302246}
      ],
      error: 149.0388957698171,
    }

    # IO.inspect(fit, label: :fit)
    expected_coefs = expected_fit[:coefs] |> coefs_nums()
    coefs = fit[:coefs] |> coefs_nums()

    # Due to the randomness in GD, these parameters will vary more than most tests
    assert coefs |> Matrex.subtract(expected_coefs) |> Matrex.sum() < 1.0e-2
  end

  defp coefs_nums(c) do
    [c |> Enum.map(& &1 |> elem(1))] |> Matrex.new()
  end
end
