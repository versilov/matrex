# Matrex

[![Build Status](https://travis-ci.org/versilov/matrex.svg?branch=master)](https://travis-ci.org/versilov/matrex)
[![hex.pm version](https://img.shields.io/hexpm/v/matrex.svg)](https://hex.pm/packages/matrex)

Fast matrix manipulation library for Elixir implemented in C native code with highly optimized CBLAS sgemm() used for matrix multiplication.

Extracted from https://github.com/sdwolfz/exlearn

## Benchmark

2015 MacBook Pro, 2.2 GHz Core i7, 16 GB RAM

```
benchmark name                iterations   average time
50x50 matrices dot product           500000   6.54 µs/op
transpose a 100x100 matrix           100000   12.40 µs/op
100x100 matrices dot product          50000   37.82 µs/op
transpose a 200x200 matrix            50000   66.05 µs/op
200x200 matrices dot product          10000   126.95 µs/op
transpose a 400x400 matrix            10000   175.82 µs/op
400x400 matrices dot product           5000   686.64 µs/op
```

## Installation

The package can be installed
by adding `matrex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:matrex, "~> 0.1"}
  ]
end
```
On MacOS everything works out of the box, thanks to Accelerate framework.

On Ubuntu you need to install scientific libraries for this package to compile:

```
> sudo apt-get install libblas-dev
```
## Matrices

For the sake of efficiency matrices are stored as binaries, which are actually arrays of floats. The first two elements of matrix binary are unsinged integer numbers of rows and columns, the rest of the array is the matrix body.

Here is a 3x3 matrix of ones:

```elixir
iex> Matrex.ones(3)
<<3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0,
  128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0,
  128, 63>>
```

You can convert to more common list of lists at any time:
```elixir
iex> Matrex.ones(3) |> Matrex.to_list_of_lists()
[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
```

## Creation

You can create new matrix in many different ways.

#### new()
From list of lists:
```elixir
iex> Matrex.new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) |> Matrex.inspect()
Rows: 3 Columns: 3
1 2 3
4 5 6
7 8 9
```

From function:
```elixir
iex> Matrex.new(3, 3, fn -> :rand.uniform() end) |> Matrex.inspect()
Rows: 3 Columns: 3
0.9895256161689758 0.4284432828426361 0.34333840012550354
0.12773089110851288 0.06270553171634674 0.5848795175552368
0.6484498977661133 0.49514150619506836 0.9283488988876343

iex> Matrex.new(3, 5, fn row, column -> row*column end) |> Matrex.inspect()
Rows: 3 Columns: 5
0 0 0 0 0
0 1 2 3 4
0 2 4 6 8
```

Or you can use functions for creating specific types of matrices.

#### zeros()

Matrix of zeros:
```elixir
iex> Matrex.zeros(3) |> Matrex.inspect()
Rows: 3 Columns: 3
0 0 0
0 0 0
0 0 0

iex> Matrex.zeros(3, 4) |> Matrex.inspect()
Rows: 3 Columns: 4
0 0 0 0
0 0 0 0
0 0 0 0
```

#### ones()

Matrix of ones:

```elixir
iex> Matrex.ones(3) |> Matrex.inspect()
Rows: 3 Columns: 3
1 1 1
1 1 1
1 1 1

iex> Matrex.ones(3, 4) |> Matrex.inspect()
Rows: 3 Columns: 4
1 1 1 1
1 1 1 1
1 1 1 1
```

#### fill()

Create matrix filled by arbitrary value:
```elixir
iex> Matrex.fill(3, 25) |> Matrex.inspect()
Rows: 3 Columns: 3
25 25 25
25 25 25
25 25 25

iex> Matrex.fill(3, 4, 13) |> Matrex.inspect()
Rows: 3 Columns: 4
13 13 13 13
13 13 13 13
13 13 13 13
```

#### random()

Create matrix filled by random numbers in [0, 1]:
```elixir
iex> Matrex.random(3) |> Matrex.inspect()
Rows: 3 Columns: 3
0.9735986590385437 0.27254486083984375 0.6614391207695007
0.807566225528717 0.765457034111023 0.036448732018470764
0.5938393473625183 0.6574462056159973 0.6980698704719543

iex> Matrex.random(3, 4) |> Matrex.inspect()
Rows: 3 Columns: 4
0.9739038944244385 0.4025186002254486 0.13004669547080994 0.6949213743209839
0.5433793067932129 0.5758569836616516 0.4285793602466583 0.13317014276981354
0.19062970578670502 0.9134745001792908 0.7655090093612671 0.910411536693573
```

#### eye()

Create square identity matrix:

```elixir
iex> Matrex.eye(5) |> Matrex.inspect()
Rows: 5 Columns: 5
1 0 0 0 0
0 1 0 0 0
0 0 1 0 0
0 0 0 1 0
0 0 0 0 1
```

#### magic()

Or even a "magic" matrix:

```elixir
iex> Matrex.magic(5) |> Matrex.inspect()
Rows: 5 Columns: 5
16 23 5 7 14
22 4 6 13 20
3 10 12 19 21
9 11 18 25 2
15 17 24 1 8
```



## Operations

#### apply()
#### add()
#### substract()
#### substract_inverse()
#### multiply()
#### multiply_with_scalar()
#### divide()
#### dot()
#### dot_and_add()
#### dot_nt()
#### dot_tn()
#### transpose()

## Utility

#### sum()
#### max()
#### argmax()
#### first()
#### inspect()
#### size()
#### at()
#### to_list()
#### to_list_of_lists()
