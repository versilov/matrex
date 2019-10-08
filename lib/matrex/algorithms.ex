# Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
#
#
# (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
#
# Permission is granted for anyone to copy, use, or modify these
# programs and accompanying documents for purposes of research or
# education, provided this copyright notice is retained, and note is
# made of any changes that have been made.
#
# These programs and documents are distributed without any warranty,
# express or implied.  As the programs were written for research
# purposes only, they have not been tested to the degree that would be
# advisable in any important application.  All use of these programs is
# entirely at the user's own risk.
#
# [ml-class] Changes Made:
# 1) Function name and argument specifications
# 2) Output display
#
# [versilov] Changes:
# 1) Ported to Elixir

defmodule Matrex.Algorithms do
  alias Matrex.Dashboard

  @moduledoc """
  Machine learning algorithms using matrices.
  """

  # a bunch of constants for line searches
  # RHO and SIG are the constants in the Wolfe-Powell conditions
  @rho 0.01
  @sig 0.5
  # don't reevaluate within 0.1 of the limit of the current bracket
  @int 0.1
  # extrapolate maximum 3 times the current bracket
  @ext 3.0
  # maximum allowed slope ratio
  @ratio 100.0
  # max 20 function evaluations per line search
  @max 20
  @float_min 2.2251e-308

  # The reduction in function value to be expected in the first line-search.
  # The original version did the following, but we simply fix it to 1.0:
  # if max(size(length)) == 2 { red=length(2); length=length(1) } else { red=1 }
  @red 1.0

  defmodule FMinCG do
    @moduledoc false

    defstruct ~w(f fParams fX d1 d2 d3 df0 df1 df2 df3 f0 f1 f2 f3 i length limit m s x x0 z1 z2 z3 line_search_failed legend)a
  end

  @doc """
  Minimizes a continuous differentiable multivariate function.

  Ported to Elixir from Octave version, found in Andre Ng's course, (c) Carl Edward Rasmussen.

  `f` — cost function, that takes two paramteters: current version of `x` and `fParams`. For example, `lr_cost_fun/2`.

  `x` — vector of parameters, which we try to optimize,
  so that cost function returns the minimum value.

  `fParams` — this value is passed as the second parameter to the cost function.

  `length` — number of iterations to perform.

  Returns column matrix of found solutions, list of cost function values and number of iterations used.

  Starting point is given by `x` (D by 1), and the function `f`, must
  return a function value and a vector of partial derivatives. The Polack-Ribiere
  flavour of conjugate gradients is used to compute search directions,
  and a line search using quadratic and cubic polynomial approximations and the
  Wolfe-Powell stopping criteria is used together with the slope ratio method
  for guessing initial step sizes. Additionally a bunch of checks are made to
  make sure that exploration is taking place and that extrapolation will not
  be unboundedly large.
  """

  @spec fmincg(
          (Matrex.t(), any, pos_integer -> {float, Matrex.t()}),
          Matrex.t(),
          any,
          integer
        ) :: {Matrex.t(), [float], pos_integer}
  def fmincg(f, %Matrex{} = x, fParams, length)
      when is_integer(length) and is_function(f, 3) do
    # Key to the variable names being used:
    #
    #   i     counts iterations or function evaluations ("epochs")
    #   s     search_direction (a vector)
    #   f1    cost (a scalar), also f0, f2, f3
    #   df1   gradient (a vector), also df0, df2, df3
    #   d1    slope (a scalar), also d2, d3
    #   z1    point (a scalar), also z2, z3
    #
    #   m     counter for maximum function evaluations per line search

    # zero the run length counter
    i = 0

    fX = []

    # get function value and gradient
    {f1, df1} = f.(x, fParams, i)

    # search direction is steepest
    s = Matrex.neg(df1)

    # the slope
    d1 = Matrex.dot_tn(s, s, -1) |> Matrex.scalar()

    # d1 = (-s′ <*> s).scalar  # this is the slope
    # initial step is red/(|s|+1)
    z1 = @red / (1 - d1)

    %FMinCG{
      fX: fX,
      f1: f1,
      df1: df1,
      s: s,
      d1: d1,
      z1: z1,
      i: i,
      length: length,
      f: f,
      fParams: fParams,
      x: x,
      line_search_failed: false,
      legend: false
    }
    |> iteration()
  end

  defp iteration(%FMinCG{i: i, length: length, fX: fX, x: x} = data)
       when i >= length do
    if data.legend, do: IO.puts(legend(i, List.last(fX)))

    {x, fX, i}
  end

  defp iteration(
         %FMinCG{
           f: f,
           fParams: fParams,
           d1: d1,
           df1: df1,
           f1: f1,
           s: s,
           x: x,
           z1: z1
         } = data
       ) do
    data = %{
      data
      | x0: x,
        f0: f1,
        df0: df1,
        x: Matrex.add(x, s, 1.0, z1)
    }

    {f2, df2} = f.(data.x, fParams, data.i)

    d2 = Matrex.dot_tn(df2, s) |> Matrex.scalar()

    # initialize point 3 equal to point 1
    data = %{
      data
      | f2: f2,
        df2: df2,
        d2: d2,
        f3: f1,
        d3: d1,
        z3: -z1,
        m: @max,
        limit: -1
    }

    {success, data} = iteration2(data)

    data = process_result(data, success)

    iteration(%{data | i: data.i + 1})
  end

  defp legend(iterations, cost),
    do:
      "Iterations #{IO.ANSI.yellow()}#{iterations}#{IO.ANSI.reset()} | Cost: #{IO.ANSI.yellow()}#{
        cost
      }#{IO.ANSI.reset()}"

  defp process_result(data, true) do
    f1 = data.f2
    fX = data.fX ++ [f1]

    if data.legend, do: IO.write(legend(data.i, f1) <> "\r")

    s =
      Matrex.subtract(
        Matrex.dot_tn(data.df2, data.df2),
        Matrex.dot_tn(data.df1, data.df2)
      )
      |> Matrex.scalar()
      |> Kernel./(Matrex.dot_tn(data.df1, data.df1) |> Matrex.scalar())
      |> Matrex.multiply(data.s)
      |> Matrex.subtract(data.df2)

    {df1, df2} = {data.df2, data.df1}

    d2 = Matrex.dot_tn(df1, s) |> Matrex.scalar()

    {d2, s} =
      if d2 > 0 do
        s = Matrex.neg(df1)
        d2 = Matrex.dot_tn(s, s, -1) |> Matrex.scalar()
        {d2, s}
      else
        {d2, s}
      end

    z1 = data.z1 * min(@ratio, data.d1 / (d2 - @float_min))
    d1 = d2

    %FMinCG{
      data
      | d1: d1,
        d2: d2,
        df1: df1,
        df2: df2,
        f1: f1,
        fX: fX,
        s: s,
        z1: z1,
        line_search_failed: false
    }
  end

  defp process_result(data, false) do
    df1 = data.df0
    {df2, df1} = {df1, data.df2}
    s = Matrex.neg(df1)
    d1 = Matrex.dot_tn(data.s, data.s, -1) |> Matrex.scalar()
    z1 = 1 / (1 - d1)

    %FMinCG{
      data
      | x: data.x0,
        f1: data.f0,
        df1: df1,
        df2: df2,
        s: s,
        d1: d1,
        z1: z1,
        line_search_failed: true
    }
  end

  # def iteration2(%FMinCG{d1: d1, d2: d2, f1: f1, f2: f2, z1: z1} = data)
  #     when f2 > f1 + z1 * @rho * d1 or d2 > -@sig * d1,
  #     do: {false, data}
  #
  # def iteration2(%FMinCG{d2: d2, d1: d1} = data)
  #     when d2 > @sig * d1,
  #     do: {true, data}
  #
  # def iteration2(%FMinCG{m: 0} = data), do: {false, data}

  defp iteration2(%FMinCG{} = data) do
    data = tighten(data)

    cond do
      data.f2 in [:nan, :inf, :neg_inf] ->
        {false, data}

      data.f2 > data.f1 + data.z1 * @rho * data.d1 or data.d2 > -@sig * data.d1 ->
        {false, data}

      data.d2 > @sig * data.d1 ->
        # IO.puts("second #{data.d1}, #{data.d2}")
        {true, data}

      data.m == 0 ->
        # IO.puts("third")
        {false, data}

      true ->
        z2 = z2(data, data.limit)
        # IO.puts("none, z2=#{z2}")

        data = %{
          data
          | z2: z2,
            f3: data.f2,
            d3: data.d2,
            z3: -z2,
            z1: data.z1 + z2,
            x: Matrex.add(data.x, data.s, 1.0, z2)
        }

        {f2, df2} = data.f.(data.x, data.fParams, data.i)

        data = %{
          data
          | m: data.m - 1,
            f2: f2,
            df2: df2,
            d2: Matrex.dot_tn(df2, data.s) |> Matrex.scalar()
        }

        iteration2(data)
    end
  end

  defp tighten(%FMinCG{f2: f2, f1: f1, z1: z1, d1: d1, d2: d2, m: m} = data)
       when not (((f2 != :nan and f2 > f1 + z1 * @rho * d1) or d2 > -@sig * d1) and m > 0),
       do: data

  defp tighten(%FMinCG{d2: d2, d3: d3, f1: f1, f2: f2, f3: f3, z1: z1, z3: z3} = data) do
    # tighten the bracket

    data = %{data | limit: z1}

    z2 =
      try do
        if f2 > f1 do
          # quadratic fit
          z3 - 0.5 * d3 * z3 * z3 / (d3 * z3 + f2 - f3)
        else
          # cubic fit
          a = a(d2, d3, f2, f3, z3)
          b = b(d2, d3, f2, f3, z3)

          (:math.sqrt(b * b - a * d2 * z3 * z3) - b) / a
        end
      rescue
        ArithmeticError ->
          # bisect
          z3 / 2
      end

    # don't accept too close to limits
    z2 = max(min(z2, @int * z3), (1 - @int) * z3)

    # update the step
    z1 = z1 + z2
    x = Matrex.add(data.x, data.s, 1.0, z2)
    {f2, df2} = data.f.(x, data.fParams, data.i)
    m = data.m - 1
    d2 = Matrex.dot_tn(df2, data.s) |> Matrex.scalar()

    # z3 is now relative to the location of z2
    z3 = z3 - z2

    %FMinCG{data | d2: d2, m: m, f2: f2, df2: df2, x: x, z1: z1, z2: z2, z3: z3}
    |> tighten()
  end

  defp a(d2, d3, f2, f3, z3), do: 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
  defp b(d2, d3, f2, f3, z3), do: 3 * (f3 - f2) - z3 * (d3 + 2 * d2)

  defp z2(%FMinCG{d2: d2, d3: d3, f2: f2, f3: f3, z1: z1, z3: z3}, limit) do
    # make cubic extrapolation

    try do
      a = a(d2, d3, f2, f3, z3)
      b = b(d2, d3, f2, f3, z3)
      -d2 * z3 * z3 / (b + :math.sqrt(b * b - a * d2 * z3 * z3))
    rescue
      # num prob?
      ArithmeticError ->
        # extrapolate the maximum amount or bisect
        if limit < -0.5, do: z1 * (@ext - 1), else: (limit - z1) / 2
    else
      # wrong sign?
      z2 when z2 < 0 ->
        # extrapolate the maximum amount or bisect
        if limit < -0.5, do: z1 * (@ext - 1), else: (limit - z1) / 2

      # extrapolation beyond max?
      z2 when limit > -0.5 and z2 + z1 > limit ->
        # bisect
        (limit - z1) / 2

      # extrapolation beyond limit
      z2 when limit < -0.5 and z2 + z1 > z1 * @ext ->
        # set to extrapolation limit
        z1 * (@ext - 1.0)

      z2 when z2 < -z3 * @int ->
        -z3 * @int

      # too close to limit?
      z2 when limit > -0.5 and z2 < (limit - z1) * (1.0 - @int) ->
        (limit - z1) * (1.0 - @int)

      z2 ->
        z2
    end
  end

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
  @spec lr_cost_fun(Matrex.t(), {Matrex.t(), Matrex.t(), number, non_neg_integer}, pos_integer) ::
          {float, Matrex.t()}
  def lr_cost_fun(
        %Matrex{} = theta,
        {%Matrex{} = x, %Matrex{} = y, lambda, digit} = _params,
        iteration \\ 0
      )
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
            :inf -> :inf
            x -> x / m + regularization
          end).()

    grad =
      x
      |> Matrex.dot_tn(Matrex.subtract(h, y))
      |> Matrex.add(Matrex.multiply(theta, l), 1.0, lambda)
      |> Matrex.divide(m)

    if theta[:rows] == 785 do
      r = div(digit - 1, 3) * 17 + 1
      c = rem(digit - 1, 3) * 30 + 1

      j_str = j |> Matrex.element_to_string() |> String.pad_leading(20)
      iter_str = iteration |> Integer.to_string() |> String.pad_leading(3)

      IO.puts("#{IO.ANSI.home()}")

      theta[2..785]
      |> Matrex.reshape(28, 28)
      |> Dashboard.heatmap(
        digit,
        :mono256,
        at: {r, c},
        title: "[#{IO.ANSI.bright()}#{rem(digit, 10)}#{IO.ANSI.normal()} | #{iter_str} #{j_str}]"
      )
    end

    {j, grad}
  end

  @doc """
  The same cost function, implemented with  operators from `Matrex.Operators` module.

  Works 2 times slower, than standard implementation. But it's a way more readable.
  """
  def lr_cost_fun_ops(
        %Matrex{} = theta,
        {%Matrex{} = x, %Matrex{} = y, lambda} = _params,
        _iteration \\ 0
      )
      when is_number(lambda) do
    # Turn off original operators
    import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]
    import Matrex
    import Matrex.Operators

    m = y[:rows]

    h = sigmoid(x * theta)
    l = ones(size(theta)) |> set(1, 1, 0.0)

    j = (-t(y) * log(h) - t(1 - y) * log(1 - h) + lambda / 2 * t(l) * pow2(theta)) / m

    grad = (t(x) * (h - y) + (theta <|> l) * lambda) / m

    {scalar(j), grad}
  end

  @doc """
  Run logistic regression one-vs-all MNIST digits recognition in parallel.
  """
  def run_lr(iterations \\ 56, concurrency \\ 1) do
    start_timestamp = :os.timestamp()

    {x, y} =
      case Mix.env() do
        :test ->
          {Matrex.load("test/data/X.mtx.gz"), Matrex.load("test/data/Y.mtx")}

        _ ->
          Dashboard.start()
          {Matrex.load("test/data/Xtest.mtx.gz"), Matrex.load("test/data/Ytest.mtx")}
      end

    x = Matrex.concat(Matrex.ones(x[:rows], 1), x)
    theta = Matrex.zeros(x[:cols], 1)

    lambda = 0.3

    solutions =
      1..10
      |> Task.async_stream(
        fn digit ->
          y3 = Matrex.apply(y, fn val -> if(val == digit, do: 1.0, else: 0.0) end)

          {sX, fX, _i} = fmincg(&lr_cost_fun/3, theta, {x, y3, lambda, digit}, iterations)

          {digit, List.last(fX), sX}
        end,
        max_concurrency: concurrency,
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

    time_elapsed = :timer.now_diff(:os.timestamp(), start_timestamp)
    IO.puts("Time elapsed: #{time_elapsed / 1_000_000} sec.")

    accuracy
  end

  @doc """
  Function of a surface with two hills.
  """
  @spec twin_peaks(float, float) :: float
  def twin_peaks(x, y) do
    x = (x - 40) / 4
    y = (y - 40) / 4

    :math.exp(-:math.pow(:math.pow(x - 4, 2) + :math.pow(y - 4, 2), 2) / 1000) +
      :math.exp(-:math.pow(:math.pow(x + 4, 2) + :math.pow(y + 4, 2), 2) / 1000) +
      0.1 * :math.exp(-:math.pow(:math.pow(x + 4, 2) + :math.pow(y + 4, 2), 2)) +
      0.1 * :math.exp(-:math.pow(:math.pow(x - 4, 2) + :math.pow(y - 4, 2), 2))
  end

  @doc """
  Computes sigmoid gradinet for the given matrix.


      g = sigmoid(X) * (1 - sigmoid(X))
  """
  @spec sigmoid_gradient(Matrex.t()) :: Matrex.t()
  def sigmoid_gradient(%Matrex{} = z) do
    s = Matrex.apply(z, :sigmoid)

    Matrex.multiply(s, Matrex.subtract(1, s))
  end

  @doc """
  Cost function for neural network with one hidden layer.

  Does delta computation in parallel.

  Ported from Andrew Ng's course, ex4.
  """
  @spec nn_cost_fun(
          Matrex.t(),
          {pos_integer, pos_integer, pos_integer, Matrex.t(), Matrex.t(), number},
          pos_integer
        ) :: {number, Matrex.t()}
  def nn_cost_fun(
        %Matrex{} = theta,
        {input_layer_size, hidden_layer_size, num_labels, x, y, lambda} = _params,
        _iteration \\ 0
      ) do
    alias Matrex, as: M

    theta1 =
      theta[1..(hidden_layer_size * (input_layer_size + 1))]
      |> M.reshape(hidden_layer_size, input_layer_size + 1)

    theta2 =
      theta[(hidden_layer_size * (input_layer_size + 1) + 1)..theta[:rows]]
      |> M.reshape(num_labels, hidden_layer_size + 1)

    # IO.write(IO.ANSI.home())
    #
    # data_side_size = trunc(:math.sqrt(theta1[:cols]))
    #
    # theta1
    # |> Matrex.submatrix(1..theta1[:rows], 2..theta1[:cols])
    # |> visual_net({5, 5}, {data_side_size, data_side_size})
    # |> Matrex.heatmap()

    m = x[:rows]

    x = M.concat(M.ones(m, 1), x)
    a2 = M.dot_nt(theta1, x) |> M.apply(:sigmoid)
    a2 = M.concat(M.ones(1, m), a2, :rows)
    a3 = M.dot_and_apply(theta2, a2, :sigmoid)

    y_b = M.zeros(num_labels, m)
    y_b = Enum.reduce(1..m, y_b, fn i, y_b -> M.set(y_b, trunc(y[i]), i, 1) end)

    c =
      M.neg(y_b)
      |> M.multiply(M.apply(a3, :log))
      |> M.subtract(M.multiply(M.subtract(1, y_b), M.apply(M.subtract(1, a3), :log)))

    theta1_sum =
      theta1
      |> M.submatrix(1..theta1[:rows], 2..theta1[:columns])
      |> M.square()
      |> M.sum()

    theta2_sum =
      theta2
      |> M.submatrix(1..theta2[:rows], 2..theta2[:columns])
      |> M.square()
      |> M.sum()

    reg = lambda / (2 * m) * (theta1_sum + theta2_sum)

    sum_c = M.sum(c)

    # Check for special sum_C value
    sum_c =
      if sum_c == :inf or sum_c == :nan do
        IO.inspect(sum_c, label: "Bad sum from a matrix")
        IO.inspect(c)
        1_000_000_000
      else
        sum_c
      end

    j = sum_c / m + reg

    # Compute gradients
    classes = M.reshape(1..num_labels, num_labels, 1)

    delta1_init = M.zeros(M.size(theta1))
    delta2_init = M.zeros(M.size(theta2))

    n_chunks = 5
    chunk_size = trunc(m / n_chunks)

    {delta1, delta2} =
      1..n_chunks
      |> Task.async_stream(fn n ->
        ((n - 1) * chunk_size + 1)..(n * chunk_size)
        |> Enum.reduce({delta1_init, delta2_init}, fn t, {delta1, delta2} ->
          a1 = M.transpose(x[t])
          z2 = M.dot(theta1, a1)
          a2 = M.concat(M.new([[1]]), M.apply(z2, :sigmoid), :rows)

          a3 = M.dot_and_apply(theta2, a2, :sigmoid)

          sigma3 = M.subtract(a3, M.apply(classes, &if(&1 == y[t], do: 1.0, else: 0.0)))

          sigma2 =
            theta2
            |> M.submatrix(1..theta2[:rows], 2..theta2[:cols])
            |> M.dot_tn(sigma3)
            |> M.multiply(sigmoid_gradient(z2))

          delta2 = M.add(delta2, M.dot_nt(sigma3, a2))
          delta1 = M.add(delta1, M.dot_nt(sigma2, a1))
          {delta1, delta2}
        end)
      end)
      |> Enum.reduce({delta1_init, delta2_init}, fn {:ok, {delta1, delta2}},
                                                    {delta1_result, delta2_result} ->
        {M.add(delta1_result, delta1), M.add(delta2_result, delta2)}
      end)

    theta1 = M.set_column(theta1, 1, M.zeros(hidden_layer_size, 1))
    theta2 = M.set_column(theta2, 1, M.zeros(num_labels, 1))
    theta1_grad = M.divide(delta1, m) |> M.add(M.multiply(lambda / m, theta1))
    theta2_grad = M.divide(delta2, m) |> M.add(M.multiply(lambda / m, theta2))

    theta = M.concat(M.to_row(theta1_grad), M.to_row(theta2_grad)) |> M.transpose()

    {j, theta}
  end

  @doc """
  Predict labels for the featurex with pre-trained neuron coefficients theta1 and theta2.
  """
  @spec nn_predict(Matrex.t(), Matrex.t(), Matrex.t()) :: Matrex.t()
  def nn_predict(theta1, theta2, x) do
    m = x[:rows]

    h1 =
      Matrex.concat(Matrex.ones(m, 1), x)
      |> Matrex.dot_nt(theta1)
      |> Matrex.apply(:sigmoid)

    Matrex.concat(Matrex.ones(m, 1), h1)
    |> Matrex.dot_nt(theta2)
    |> Matrex.apply(:sigmoid)
  end

  # Reshape each row of theta into a n_rows x n_cols matrix
  # Group these matrices into a rows x cols big matrix for visualization
  defp visual_net(theta, {rows, cols} = _visu_size, {n_rows, n_cols} = _neuron_size) do
    1..theta[:rows]
    |> Enum.map(&(theta[&1] |> Matrex.reshape(n_rows, n_cols)))
    |> Matrex.reshape(rows, cols)
  end

  @sample_side_size 20
  @input_layer_size @sample_side_size * @sample_side_size
  @hidden_layer_size 25
  @num_labels 10

  @doc """
  Run neural network with one hidden layer.


  """
  def run_nn(epsilon \\ 0.12, iterations \\ 100, lambdas \\ [0.1, 5, 50]) do
    start_timestamp = :os.timestamp()

    x = Matrex.load("test/data/X.mtx.gz")
    y = Matrex.load("test/data/Y.mtx")

    # {x_train, y_train, x_test, y_test} = split_data(x, y)

    {x_train, y_train, x_test, y_test} = {x, y, x, y}

    lambdas
    |> Task.async_stream(
      fn lambda ->
        initial_theta1 =
          random_weights(@input_layer_size, @hidden_layer_size, epsilon) |> Matrex.to_row()

        initial_theta2 =
          random_weights(@hidden_layer_size, @num_labels, epsilon) |> Matrex.to_row()

        initial_nn_params = Matrex.concat(initial_theta1, initial_theta2) |> Matrex.transpose()

        {sX, fX, _i} =
          fmincg(
            &nn_cost_fun/3,
            initial_nn_params,
            {@input_layer_size, @hidden_layer_size, @num_labels, x_train, y_train, lambda},
            iterations
          )

        {lambda, List.last(fX), sX}
      end,
      timeout: 600_000,
      max_concurrency: 8
    )
    |> Enum.each(fn
      {:ok, {lambda, cost, sX}} ->
        # Unpack thetas from the found solution
        theta1 =
          sX[1..(@hidden_layer_size * (@input_layer_size + 1))]
          |> Matrex.reshape(@hidden_layer_size, @input_layer_size + 1)

        theta2 =
          sX[(@hidden_layer_size * (@input_layer_size + 1) + 1)..sX[:rows]]
          |> Matrex.reshape(@num_labels, @hidden_layer_size + 1)

        predictions = Matrex.Algorithms.nn_predict(theta1, theta2, x_test)

        1..predictions[:rows]
        |> Enum.reduce(0, fn row, acc ->
          if y_test[row] == predictions[row][:argmax] do
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
        |> IO.inspect(label: "\rTraining set accuracy with lambda #{lambda} and cost #{cost}")

        theta1
        |> Matrex.submatrix(1..theta1[:rows], 2..theta1[:cols])
        |> visual_net({5, 5}, {@sample_side_size, @sample_side_size})
        |> Matrex.heatmap()

      _ ->
        :noop
    end)

    time_elapsed = :timer.now_diff(:os.timestamp(), start_timestamp)
    IO.puts("Time elapsed: #{time_elapsed / 1_000_000} sec.")
  end

  defp random_weights(l_in, l_out, epsilon) do
    Matrex.random(l_out, 1 + l_in)
    |> Matrex.multiply(2 * epsilon)
    |> Matrex.subtract(epsilon)
  end
end
