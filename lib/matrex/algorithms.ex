defmodule Matrex.Algorithms do
  @moduledoc """
  Contains machine learning algorithms using matrices.
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
    defstruct ~w(f fParams fX d1 d2 d3 df0 df1 df2 df3 f0 f1 f2 f3 i length limit m s x x0 z1 z2 z3 line_search_failed)a
  end

  @doc """
  Minimizes a continuous differentiable multivariate function.

  """

  @spec fmincg(
          (Matrex.t(), any -> {float, Matrex.t()}),
          Matrex.t(),
          any,
          integer
        ) :: {Matrex.t(), [float], pos_integer}
  def fmincg(f, %Matrex{} = x, fParams, length)
      when is_integer(length) and is_function(f, 2) do
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
    {f1, df1} = f.(x, fParams)

    # search direction is steepest
    s = Matrex.neg(df1)

    # the slope
    d1 =
      s
      |> Matrex.multiply(-1)
      |> Matrex.dot_tn(s)
      |> Matrex.first()

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
      line_search_failed: false
    }
    |> iteration()
  end

  def iteration(%FMinCG{i: i, length: length, fX: fX, x: x})
      when i >= length do
    IO.puts("Iterations #{i} | Cost: #{List.last(fX)}")

    {x, fX, i}
  end

  def iteration(
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
        x: Matrex.add(x, Matrex.multiply(z1, s))
    }

    {f2, df2} = f.(data.x, fParams)

    d2 = Matrex.dot_tn(df2, s) |> Matrex.first()

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

  def process_result(data, true) do
    f1 = data.f2
    fX = data.fX ++ [f1]

    IO.write("Iteration #{data.i} | Cost: #{f1}\r")

    s =
      Matrex.substract(
        Matrex.dot_tn(data.df2, data.df2),
        Matrex.dot_tn(data.df1, data.df2)
      )
      |> Matrex.first()
      |> Kernel./(Matrex.dot_tn(data.df1, data.df1) |> Matrex.first())
      |> Matrex.multiply(data.s)
      |> Matrex.substract(data.df2)

    {df1, df2} = {data.df2, data.df1}

    d2 = Matrex.dot_tn(df1, s) |> Matrex.first()

    {d2, s} =
      if d2 > 0 do
        s = Matrex.neg(df1)
        d2 = Matrex.dot_tn(Matrex.neg(s), s) |> Matrex.first()
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

  def process_result(data, false) do
    df1 = data.df0
    {df2, df1} = {df1, data.df2}
    s = Matrex.neg(df1)
    d1 = Matrex.dot_tn(Matrex.neg(data.s), data.s) |> Matrex.first()
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

  def iteration2(%FMinCG{} = data) do
    data = tighten(data)

    cond do
      data.f2 in [NaN, Inf, NegInf] ->
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
            x: Matrex.add(data.x, Matrex.multiply(data.s, z2))
        }

        {f2, df2} = data.f.(data.x, data.fParams)

        data = %{
          data
          | m: data.m - 1,
            f2: f2,
            df2: df2,
            d2: Matrex.dot_tn(df2, data.s) |> Matrex.first()
        }

        iteration2(data)
    end
  end

  def tighten(%FMinCG{f2: f2, f1: f1, z1: z1, d1: d1, d2: d2, m: m} = data)
      when not (((f2 != NaN and f2 > f1 + z1 * @rho * d1) or d2 > -@sig * d1) and m > 0),
      do: data

  def tighten(%FMinCG{d2: d2, d3: d3, f1: f1, f2: f2, f3: f3, z1: z1, z3: z3} = data) do
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
    x = Matrex.add(data.x, Matrex.multiply(z2, data.s))
    {f2, df2} = data.f.(x, data.fParams)
    m = data.m - 1
    d2 = Matrex.dot_tn(df2, data.s) |> Matrex.first()

    # z3 is now relative to the location of z2
    z3 = z3 - z2

    %FMinCG{data | d2: d2, m: m, f2: f2, df2: df2, x: x, z1: z1, z2: z2, z3: z3}
    |> tighten()
  end

  def a(d2, d3, f2, f3, z3), do: 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
  def b(d2, d3, f2, f3, z3), do: 3 * (f3 - f2) - z3 * (d3 + 2 * d2)

  def z2(%FMinCG{d2: d2, d3: d3, f2: f2, f3: f3, z1: z1, z3: z3}, limit) do
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
end
