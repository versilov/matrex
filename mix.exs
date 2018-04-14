defmodule Mix.Tasks.Compile.Matrex do
  def run(_) do
    File.mkdir("priv")
    {exec, args} = {"make", []}

    if System.find_executable(exec) do
      build(exec, args)
      Mix.Project.build_structure()
      :ok
    else
      nocompiler_error(exec)
    end
  end

  def build(exec, args) do
    {result, error_code} = System.cmd(exec, args, stderr_to_stdout: true)
    IO.binwrite(result)
    if error_code != 0, do: build_error(exec)
  end

  defp nocompiler_error(exec) do
    raise Mix.Error, message: nocompiler_message(exec)
  end

  defp build_error(exec) do
    raise Mix.Error, message: build_message(exec)
  end

  defp nocompiler_message(exec) do
    """
    Could not find the compiler program `#{exec}`.
    """
  end

  defp build_message(exec) do
    """
    Could not build the program with `#{exec}`.
    """
  end
end

defmodule Matrex.MixProject do
  use Mix.Project

  def project do
    [
      app: :matrex,
      version: "0.1.0",
      elixir: "~> 1.6",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      compilers: [:matrex] ++ Mix.compilers(),
      description:
        "Fast matrix manipulation library for Elixir with native C implementation using CBLAS.",
      dialyzer: [
        flags: [
          # --------------------------
          # Flags that DISABLE checks
          # --------------------------
          "-Wno_return",
          "-Wno_unused",
          # "-Wno_improper_lists",
          # "-Wno_fun_app",
          "-Wno_match",
          # "-Wno_opaque",
          "-Wno_fail_call",
          "-Wno_contracts",
          "-Wno_behaviours",
          # "-Wno_missing_calls",
          # "-Wno_undefined_callbacks",
          # -------------------------
          # Flags that ENABLE checks
          # ------------------------
          "-Wunmatched_returns",
          "-Werror_handling",
          "-Wrace_conditions",
          # "-Wunderspecs",
          "-Wunknown"
          # "-Woverspecs",
          # "-Wspecdiffs"
        ]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:benchfella, "0.3.4", only: :dev},
      {:dialyxir, "0.5.0", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "native",
        "Makefile",
        "README.md",
        "mix.exs"
      ],
      maintainers: [
        "Stas Versilov"
      ],
      licenses: ["simplified BSD"],
      build_tools: ["make"],
      links: %{
        "GitHub" => "https://github.com/versilov/matrex"
      }
    ]
  end
end
