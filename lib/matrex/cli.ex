defmodule Matrex.CLI do
  def iex_env do
    alias Matrex, as: M
    alias Matrex.Algorithms
    __ENV__
  end

  def main(args \\ []) do
    process_args(args)
    IEx.Server.run(prefix: "matrex", env: iex_env())
  end

  defp process_args([]), do: :noop
  defp process_args([filename]), do: Code.compile_file(filename)
end
