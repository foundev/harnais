# Harnais

Harnais is a lightweight agentic harness written in Go, similar to pi or
Claude Code but smaller in scope and more human sized.

It runs an agent loop against a model backend — Anthropic's Messages API
by default, any model on OpenRouter, the DeepSeek / OpenAI APIs (all via an
OpenAI-compatible endpoint), or your ChatGPT subscription by reusing the
Codex CLI login. The model can call exactly four tools — `bash`, `read`,
`edit`, `write` — and the harness executes them locally, feeds the results
back, and repeats until the model gives a final answer. The implementation
uses only the Go standard library.

## Requirements

- Go 1.24 or newer (module targets `go 1.27.1`)
- A credential for your backend: `ANTHROPIC_API_KEY` (default),
  `OPENROUTER_API_KEY` with `-provider=openrouter`, `DEEPSEEK_API_KEY`
  with `-provider=deepseek`, `OPENAI_API_KEY` with `-provider=openai`,
  or a Codex CLI login (`codex login`) with `-provider=codex`.
  Interactive sessions open without any credential — one is only needed
  when you send the first prompt or switch backends with `/provider`.

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

Or bill your ChatGPT subscription through the Codex login you already have —
no API key needed. Auth is Codex's job (`codex login` / `codex logout`);
harnais only reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`),
refreshing the access token the same way Codex CLI does:

```sh
harnais -provider=codex "explain what main.go does"
```

Start an interactive session (history is kept until `/reset` or `/exit`).
The prompt shows the active backend:

```sh
harnais
anthropic> explain what main.go does
anthropic> add retry logic to the API client
```

The backend is the saved `/provider` choice (`anthropic` on first run).
One-shot mode uses the saved backend too; `-provider` overrides it for one
run, and `-p` takes the prompt as a flag so other flags can follow it:
Session commands:

```sh
anthropic> /provider openrouter   # switch backend, saved as default (model resets, history kept)
anthropic> /model my-model        # use a different model id, saved as default
anthropic> /effort high           # reasoning effort (low, medium, high, default), saved as default
anthropic> /reset                 # clear history
anthropic> /exit                  # leave
```

Provider, model, and effort persist in `$XDG_CONFIG_HOME/harnais/config.json`
(`~/.config/harnais/config.json` by default), so the next launch —
session or one-shot — starts with the same triple. `-model` and `-effort`
override the saved values for one run without changing them.

`/effort` is sent on the wire to the `codex` (`reasoning.effort`),
`openai` and `openrouter` (`reasoning_effort`) backends. Anthropic has no
equivalent setting and DeepSeek always reasons, so the command has no
effect there. `-effort` sets it for one-shot mode too.

| Flag           | Default                        | Meaning                                        |
| -------------- | ------------------------------ | ---------------------------------------------- |
| `-provider`    | saved (`anthropic` first run)  | Backend for this run, without changing the saved default |
| `-p`           | `""`                           | One non-interactive prompt and exit (so flags can follow the prompt) |
| `-model`       | saved, else per-provider (see below) | Model ID for this run (`ANTHROPIC_MODEL` overrides the default) |
| `-key`         | per-provider env var           | API key                                        |
| `-effort`      | saved, else `""`               | Reasoning effort for this run: `low`, `medium`, `high` (codex, openai, openrouter) |
| `-n`           | `25`                           | Max agent iterations per prompt                |
| `-max-tokens`  | `8192`                         | Max tokens per model response                  |
| `-y`           | `false`                        | Run mutating tools without asking              |
| `-no-sandbox`  | `false`                        | Run `bash` without the OS sandbox              |
| `-system`      | `""`                           | Extra system instructions                      |
| `-version`     | —                              | Print version and exit                         |

Default models: `claude-sonnet-4-20250514` for `-provider=anthropic`,
`anthropic/claude-sonnet-4.5` for `-provider=openrouter`, `deepseek-chat`
for `-provider=deepseek`, `gpt-5-mini` for `-provider=openai`,
`gpt-5.3-codex` for `-provider=codex`.

| Environment          | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `ANTHROPIC_API_KEY`  | API key for the anthropic backend                    |
| `ANTHROPIC_MODEL`    | Overrides the default `-model` value (either provider) |
| `ANTHROPIC_BASE_URL` | Anthropic API root (default `https://api.anthropic.com`) |
| `OPENROUTER_API_KEY` | API key for `-provider=openrouter`                   |
| `OPENROUTER_BASE_URL`| OpenRouter API root (default `https://openrouter.ai/api/v1`) |
| `OPENROUTER_REFERER` | Optional `HTTP-Referer` header sent to OpenRouter    |
| `DEEPSEEK_API_KEY`   | API key for `-provider=deepseek`                     |
| `DEEPSEEK_BASE_URL`  | DeepSeek API root (default `https://api.deepseek.com`) |
| `OPENAI_API_KEY`     | API key for `-provider=openai`                       |
| `OPENAI_BASE_URL`    | OpenAI API root (default `https://api.openai.com/v1`) |
| `CODEX_HOME`         | Directory holding Codex `auth.json` (default `~/.codex`) |
| `CODEX_BASE_URL`     | Codex backend root (default `https://chatgpt.com/backend-api/codex`) |
| `HARNAIS_DEBUG`      | When set to a file path, appends Codex request/response bodies there (never tokens) |

In a session, `/help` lists the slash commands (`/provider`, `/model`, `/effort`, `/reset`, `/exit`).

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

## Sandbox

`bash` runs under an OS sandbox by default (macOS Seatbelt, modeled on
[Codex's sandbox](https://github.com/openai/codex/tree/main/codex-rs/sandboxing)):
deny-by-default, writes confined to the command workdir plus standard
temp/cache locations, no network. Reads stay open and `read`/`edit`/`write`
are unaffected (path-confined and still confirmed). The agent cannot switch
sandboxing off per-call — only the `--no-sandbox` process flag escapes, so
prompt injection can't break out that way. Unsandboxed runs are marked
`[no sandbox]` in the progress output, and sandbox denials name the escape
hatch. Other platforms currently run unsandboxed with a one-time warning
(Linux confinement is the follow-up).

## Layout

- `main.go` — CLI flags, one-shot mode, interactive REPL
- `agent.go` — the agentic tool-use loop and system prompt
- `anthropic.go` — minimal Messages API client (`net/http` + `encoding/json`)
- `openai.go` — OpenAI-compatible client (OpenRouter, DeepSeek, OpenAI) translated onto the same loop
- `codex.go` — Codex subscription backend: reuses the Codex CLI login, speaks the Responses API
- `tools.go` — the four tool implementations
- `tools_test.go` — tool tests (`go test ./...`)

## Roadmap

- Linux sandboxing (bubblewrap/Landlock, mirroring Codex)
- MCP support (Model Context Protocol)
- ACP support (Agent Client Protocol)
- Remote access via a web server
- Integrated web search tooling
- Image support
- @ files to add them to the context
- Z.ai subscription support

## License

MIT — see [LICENSE](LICENSE).
