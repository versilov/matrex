random_numexy = fn size ->
  for(
    _ <- 1..size,
    do:
      for(
        _ <- 1..size,
        do: :rand.uniform()
      )
  )
  |> Numexy.new()
end

trans_jobs = %{
  "Matrex" =>
    {fn m -> Matrex.transpose(m) end, before_scenario: fn size -> Matrex.random(size) end},
  "ExMatrix" =>
    {fn m ->
       ExMatrix.transpose(m)
     end,
     before_scenario: fn size ->
       ExMatrix.random_cells(size, size, 100)
     end},
  "Matrix" =>
    {fn m -> Matrix.transpose(m) end, before_scenario: fn size -> Matrix.rand(size, size) end},
  "Numexy" => {fn m -> Numexy.transpose(m) end, before_scenario: random_numexy}
}

trans_inputs = [
  {"100x100", 100},
  {"1000x1000", 1000},
  {"5000x5000", 5000}
]

Benchee.run(
  trans_jobs,
  inputs: trans_inputs,
  formatters: [
    &Benchee.Formatters.HTML.output/1,
    &Benchee.Formatters.Console.output/1
  ],
  formatter_options: [
    html: [
      file: Path.expand("output/transpose.html", __DIR__)
    ]
  ]
)
