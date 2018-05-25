defmodule Matrex.Dashboard do
  @moduledoc false

  use GenServer
  # API
  def start do
    IO.write("#{IO.ANSI.reset()}#{IO.ANSI.clear()}")
    GenServer.start_link(__MODULE__, %{frames: 0, cells: %{}}, name: __MODULE__)
  end

  def heatmap(%Matrex{} = m, cell_id, type, opts) do
    GenServer.cast(__MODULE__, {:heatmap, {cell_id, m, type, opts}})
  end

  # Callbacks
  @impl true
  def init(state) do
    schedule_work(1)
    {:ok, state}
  end

  @impl true
  def handle_cast({:heatmap, {cell_id, matrex, type, opts}}, state) do
    {:noreply, put_in(state, [:cells, cell_id], {matrex, type, opts})}
  end

  @impl true
  def handle_info(:print, %{frames: _f, cells: cells} = state) when cells == %{} do
    # No work. Wait a bit and check one more time.
    schedule_work(1)
    {:noreply, state}
  end

  @impl true
  def handle_info(:print, %{frames: f, cells: cells}) do
    cells
    |> Map.values()
    |> Enum.each(fn {matrex, type, opts} ->
      Matrex.heatmap(matrex, type, opts)
    end)

    # Reschedule immediately once more
    schedule_work(1)
    {:noreply, %{frames: f + 1, cells: %{}}}
  end

  defp schedule_work(millis) do
    Process.send_after(self(), :print, millis)
  end
end
