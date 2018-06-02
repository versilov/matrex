defmodule Matrex.Array.NIFs do
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

    :ok = :erlang.load_nif(:filename.join(priv_dir, "array_nifs"), 0)
  end

  def add_arrays(data1, data2, type) when is_binary(data1) and is_binary(data2) and is_atom(type),
    do: :erlang.nif_error(:nif_library_not_loaded)
end
