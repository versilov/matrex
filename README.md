# Matrex

Matrix manipulation library for Elixir implemented in C native code.
Extracted from https://github.com/sdwolfz/exlearn

## Matrex Bench on 2015 MacBook Pro

benchmark name                iterations   average time
50x50 matrices dot product        500000   6.89 µs/op
transpose a 100x100 matrix        100000   28.55 µs/op
100x100 matrices dot product       50000   37.50 µs/op
200x200 matrices dot product       10000   124.31 µs/op
transpose a 200x200 matrix         10000   126.82 µs/op
transpose a 400x400 matrix          5000   441.10 µs/op
400x400 matrices dot product        5000   673.23 µs/op

## Installation

The package can be installed
by adding `matrex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:matrex, "~> 0.1.0"}
  ]
end
```
