defmodule Matrex.Dashboard do
  use GenServer
  # API
  def start do
    IO.write("#{IO.ANSI.reset()}#{IO.ANSI.clear()}")
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def heatmap(%Matrex{} = m, cell_id, type, opts) do
    GenServer.cast(__MODULE__, {:heatmap, {cell_id, m, type, opts}})
  end

  # Callbacks
  @impl true
  def init(state) do
    schedule_work()
    {:ok, state}
  end

  @impl true
  def handle_cast({:heatmap, {cell_id, matrex, type, opts}}, state) do
    {:noreply, Map.put(state, cell_id, {matrex, type, opts})}
  end

  @impl true
  def handle_info(:print, state) do
    state
    |> Map.values()
    |> Enum.each(fn {matrex, type, opts} ->
      Matrex.heatmap(matrex, type, opts)
    end)

    # Reschedule once more
    schedule_work()
    {:noreply, %{}}
  end

  defp schedule_work() do
    # In 1 second
    Process.send_after(self(), :print, 1)
  end
end
