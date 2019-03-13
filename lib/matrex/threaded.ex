defmodule Matrex.Threaded do
  @moduledoc false

  # Matrix functions that do their work in several parallel threads for the sake of speed.

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

    case :erlang.load_nif(:filename.join(priv_dir, "matrix_threaded_nifs"), 0) do
      :ok ->
        :ok

      {:error, {:load_failed, reason}} ->
        IO.warn("Error loading NIF #{reason}")
        :ok
    end
  end

  def apply_math(matrix, function)
      when is_binary(matrix) and is_atom(function) do
    :erlang.nif_error(:nif_library_not_loaded)
  end
end
