random_numexy = fn size ->
  1..size
  |> Enum.map(fn _ ->
    for _ <- 1..size, do: :rand.uniform()
  end)
  |> Numexy.new()
end

dot_jobs = %{
  "Matrex" =>
    {fn {a, b} -> Matrex.dot(a, b) end,
     before_scenario: fn size -> {Matrex.random(size), Matrex.random(size)} end},
  "ExMatrix" =>
    {fn {a, b} ->
       ExMatrix.multiply(a, b)
     end,
     before_scenario: fn size ->
       {ExMatrix.random_cells(size, size, 100), ExMatrix.random_cells(size, size, 100)}
     end},
  "Matrix" =>
    {fn {a, b} -> Matrix.mult(a, b) end,
     before_scenario: fn size -> {Matrix.rand(size, size), Matrix.rand(size, size)} end},
  "Numexy" =>
    {fn {a, b} -> Numexy.dot(a, b) end,
     before_scenario: fn size ->
       {random_numexy.(size), random_numexy.(size)}
     end}
}

dot_inputs = [
  {"50x50", 50},
  {"100x100", 100},
  {"500x500", 500}
]

Benchee.run(
  dot_jobs,
  parallel: 1,
  warmup: 2,
  time: 5,
  inputs: dot_inputs,
  formatters: [
    &Benchee.Formatters.HTML.output/1,
    &Benchee.Formatters.Console.output/1
  ],
  formatter_options: [
    html: [
      file: Path.expand("output/dot.html", __DIR__)
    ]
  ]
)
