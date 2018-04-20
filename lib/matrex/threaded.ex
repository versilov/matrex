defmodule Matrex.Threaded do
  @moduledoc """
  Matrix functions that do their work in several parallel threads for the sake of speed.
  """

  @on_load :load_nifs

  @doc false
  @spec load_nifs :: :ok
  def load_nifs do
    priv_dir =
      case :code.priv_dir(__MODULE__) do
        {:error, _} ->
          ebin_dir = :code.which(__MODULE__) |> :filename.dirname()
          app_path = :filename.dirname(ebin_dir)
          :filename.join(app_path, "priv")

        path ->
          path
      end

    :ok = :erlang.load_nif(:filename.join(priv_dir, "matrix_threaded_nifs"), 0)
  end

  @spec tadd(binary, binary) :: binary
  def tadd(first, second)
      when is_binary(first) and is_binary(second) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end

  def apply_exp(rows, cols)
      when is_integer(rows) and is_integer(cols) do
    # excoveralls ignore
    :erlang.nif_error(:nif_library_not_loaded)

    # excoveralls ignore
    random_size = :rand.uniform(2)
    # excoveralls ignore
    <<1::size(random_size)>>
  end
end
