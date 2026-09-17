# Harnais

Harnais is a lightweight agentic harness written in Go, similar to pi or
Claude Code but smaller in scope and more human sized.

It runs an agent loop against a model backend — Anthropic's Messages API
by default, any model on OpenRouter, or the DeepSeek / OpenAI APIs (all via
an OpenAI-compatible endpoint). The model can call exactly four tools —
`bash`, `read`, `edit`, `write` — and the harness executes them locally,
feeds the results back, and repeats until the model gives a final answer.
The implementation uses only the Go standard library.

## Requirements

- Go 1.24 or newer (module targets `go 1.27.1`)
- An API key for your backend: `ANTHROPIC_API_KEY` (default),
  `OPENROUTER_API_KEY` with `-provider=openrouter`, `DEEPSEEK_API_KEY`
  with `-provider=deepseek`, or `OPENAI_API_KEY` with `-provider=openai`

## Build

```sh
go build -o harnais .
```

## Usage

Ask once and print the answer:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
harnais "list all TODO comments in this repo and fix the trivial ones"
```

Use any model on OpenRouter instead:

```sh
export OPENROUTER_API_KEY=sk-or-...
harnais -provider=openrouter "explain what main.go does"
harnais -provider=openrouter -model "openai/gpt-5-mini" "fix the failing test"
```

Or the DeepSeek API directly:

```sh
export DEEPSEEK_API_KEY=sk-...
harnais -provider=deepseek "explain what main.go does"
```

Or OpenAI directly:

```sh
export OPENAI_API_KEY=sk-...
harnais -provider=openai "explain what main.go does"
```

Start an interactive session (history is kept until `/reset` or `/exit`):

```sh
harnais
> explain what main.go does
> add retry logic to the API client
```

| Flag           | Default                        | Meaning                                        |
| -------------- | ------------------------------ | ---------------------------------------------- |
| `-provider`    | `anthropic`                    | Backend: `anthropic`, `openrouter`, `deepseek` or `openai` (`HARNAIS_PROVIDER` overrides the default) |
| `-model`       | per-provider (see below)       | Model ID (`ANTHROPIC_MODEL` overrides the default) |
| `-key`         | per-provider env var           | API key                                        |
| `-n`           | `25`                           | Max agent iterations per prompt                |
| `-max-tokens`  | `8192`                         | Max tokens per model response                  |
| `-y`           | `false`                        | Run mutating tools without asking              |
| `-system`      | `""`                           | Extra system instructions                      |
| `-version`     | —                              | Print version and exit                         |

Default models: `claude-sonnet-4-20250514` for `-provider=anthropic`,
`anthropic/claude-sonnet-4.5` for `-provider=openrouter`, `deepseek-chat`
for `-provider=deepseek`, `gpt-5-mini` for `-provider=openai`.

| Environment          | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `HARNAIS_PROVIDER`   | Overrides the default `-provider` value              |
| `ANTHROPIC_API_KEY`  | API key for `-provider=anthropic`                    |
| `ANTHROPIC_MODEL`    | Overrides the default `-model` value (either provider) |
| `ANTHROPIC_BASE_URL` | Anthropic API root (default `https://api.anthropic.com`) |
| `OPENROUTER_API_KEY` | API key for `-provider=openrouter`                   |
| `OPENROUTER_BASE_URL`| OpenRouter API root (default `https://openrouter.ai/api/v1`) |
| `OPENROUTER_REFERER` | Optional `HTTP-Referer` header sent to OpenRouter    |
| `DEEPSEEK_API_KEY`   | API key for `-provider=deepseek`                     |
| `DEEPSEEK_BASE_URL`  | DeepSeek API root (default `https://api.deepseek.com`) |
| `OPENAI_API_KEY`     | API key for `-provider=openai`                       |
| `OPENAI_BASE_URL`    | OpenAI API root (default `https://api.openai.com/v1`) |

Interactive commands: `/help`, `/reset`, `/exit`.

By default the harness asks before running any mutating tool (`bash`,
`edit`, `write`); pass `-y` to skip confirmations. Tool progress goes to
stderr, so `harnais "prompt" > answer.txt` captures just the final answer.

## Tools

| Tool    | Input                                    | Effect                                                        |
| ------- | ---------------------------------------- | ------------------------------------------------------------- |
| `bash`  | `command`, `workdir?`, `timeout_ms?`     | Runs `sh -c`, returns combined output, exit status, duration  |
| `read`  | `path`, `offset?`, `limit?`              | Returns a 1-based line-numbered file window                   |
| `edit`  | `path`, `find`, `replace`                | Replaces one exact match; fails unless the match is unique    |
| `write` | `path`, `content`                        | Creates or overwrites a file, making parent directories       |

There is deliberately no sandboxing yet: the agent runs shell commands as
your user with your environment. Review proposed commands (or run with `-y`
only in a disposable checkout).

## Layout

- `main.go` — CLI flags, one-shot mode, interactive REPL
- `agent.go` — the agentic tool-use loop and system prompt
- `anthropic.go` — minimal Messages API client (`net/http` + `encoding/json`)
- `openai.go` — OpenAI-compatible client (OpenRouter, DeepSeek, OpenAI) translated onto the same loop
- `tools.go` — the four tool implementations
- `tools_test.go` — tool tests (`go test ./...`)

## Roadmap

- sandboxing
- MCP support (Model Context Protocol)
- ACP support (Agent Client Protocol)
- Remote access via a web server
- Integrated web search tooling
- Image support
- @ files to add them to the context
- Z.ai subscription support
- Codex subscription support

## License

MIT — see [LICENSE](LICENSE).
