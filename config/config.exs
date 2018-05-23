use Mix.Config

if Mix.env() == :dev do
  config :mix_test_watch,
    extra_extensions: [
      ".c"
    ]
end
