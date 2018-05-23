use Mix.Config

if Mix.env() == :dev do
  config :mix_test_watch,
    clear: true,
    extra_extensions: [
      ".c"
    ]
end
