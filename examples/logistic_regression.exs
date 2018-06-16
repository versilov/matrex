defmodule LogisticRegression do
  alias Matrex.Array
  use ExUnit.Case

  @doc """
  Linear regression cost and gradient function with regularization from Andrew Ng's course (ex3).

  Computes the cost of using `theta` as the parameter for regularized logistic regression and the
  gradient of the cost w.r.t. to the parameters.

  Compatible with `fmincg/4` algorithm from thise module.

  `theta`  — parameters, to compute cost for

  `X`  — training data input.

  `y`  — training data output.

  `lambda`  — regularization parameter.

  """
  @spec lr_cost_fun(Array.t(), {Array.t(), Array.t(), number, non_neg_integer}, pos_integer) ::
          {float, Array.t()}
  def lr_cost_fun(
        %Array{} = theta,
        {%Array{} = x, %Array{} = y, lambda, _digit} = _params,
        _iteration \\ 0
      )
      when is_number(lambda) do

    alias Array, as: A
    m = y[:rows]

    h =
      A.dot(x, theta)
      |> A.apply(:sigmoid)

    l = A.ones(theta) |> A.set({1, 1}, 0)
    
    regularization =
      l
      |> A.dot_tn(A.square(theta))
      |> A.scalar()
      |> Kernel.*(lambda / (2 * m))
    
    j =
      y
      |> A.dot_tn(A.apply(h, :log), -1)
      |> A.subtract(
        A.dot_tn(
          A.subtract(A.ones(y), y),
          A.apply(A.subtract(A.ones(h), h), :log)
        )
      )
      |> A.scalar()
      |> (fn
            :nan -> :nan
            :inf -> :inf
            x -> x / m + regularization
          end).()
    
    grad =
      x
      |> A.dot_tn(A.subtract(h, y))
    #   |> Array.add(Array.multiply(theta, l), 1.0, lambda)
    #   |> Array.divide(m)
    #
    # if theta[:rows] == 785 do
    #   r = div(digit - 1, 3) * 17 + 1
    #   c = rem(digit - 1, 3) * 30 + 1
    #
    #   j_str = j |> Matrex.element_to_string() |> String.pad_leading(20)
    #   iter_str = iteration |> Integer.to_string() |> String.pad_leading(3)
    #
    #   IO.puts("#{IO.ANSI.home()}")
    #
    #   theta[2..785]
    #   |> Matrex.reshape(28, 28)
    #   |> Dashboard.heatmap(
    #     digit,
    #     :mono256,
    #     at: {r, c},
    #     title: "[#{IO.ANSI.bright()}#{rem(digit, 10)}#{IO.ANSI.normal()} | #{iter_str} #{j_str}]"
    #   )
    # end
    #
    {j, grad}
  end

  test "#lr_cost_func() computes cost function with an Array" do
    theta_t = Array.new([-2, -1, 1, 2], {4, 1}, :float64)

    x_t =
      Array.from_string("""
        1.00000   0.10000   0.60000   1.10000
        1.00000   0.20000   0.70000   1.20000
        1.00000   0.30000   0.80000   1.30000
        1.00000   0.40000   0.90000   1.40000
        1.00000   0.50000   1.00000   1.50000
      """, :float64)

    y_t = Array.from_string("1;0;1;0;1", :float64)
    lambda_t = 3

    expected_j = 2.5348193645477295

    expected_grad =
      Array.from_string(
        "0.1465613692998886; -0.5485584139823914; 0.7247222661972046; 1.3980028629302979"
      )

    # On Travis CI the first element is computed a bit differently.
    expected_grad2 =
      Array.from_string(
        "0.14656135439872742; -0.5485584139823914; 0.7247222661972046; 1.3980028629302979"
      )

    {j, grad} = LogisticRegression.lr_cost_fun(theta_t, {x_t, y_t, lambda_t, 0}) |> IO.inspect()

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j

    {j, grad} = LogisticRegression.lr_cost_fun_ops(theta_t, {x_t, y_t, lambda_t})

    assert grad == expected_grad || grad == expected_grad2
    assert j == expected_j
  end
end
