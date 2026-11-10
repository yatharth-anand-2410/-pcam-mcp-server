# PCam inference MCP server

An MCP server that lets AI assistants (Claude Desktop, Cursor, Claude Code) use a PatchCamelyon cancer-detection inference API through three tools.

**Status: prototype.** It runs against the included mock API. The real API's endpoint paths and payloads are assumptions until its OpenAPI docs are available; they are isolated at the top of `server.py`.

**Research use only. Not a medical device; not for clinical diagnosis.** Every tool response carries this disclaimer.

## Tools

| Tool | Input | Returns |
|---|---|---|
| `classify_tissue_patch` | `file_path` or `image_base64` (exactly one) | label, confidence, model version |
| `explain_prediction` | `file_path` or `image_base64` (exactly one) | Grad-CAM heatmap as an image (upscaled to at least 384 px) plus a one-line summary |
| `get_model_info` | none | model name, version, input size, decision threshold |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest        # 14 tests, inference API mocked
```

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `PCAM_API_URL` | yes | Base URL of the inference API |
| `PCAM_API_KEY` | yes | Sent as the `X-API-Key` header |
| `PCAM_API_TIMEOUT` | no | Request timeout in seconds (default 30) |

## Try it locally with the mock API

```bash
uv run mock_api.py   # http://127.0.0.1:8765, API key "dev-key", fake predictions
```

`samples/synthetic_patch.png` is a generated image for smoke tests, not real tissue.

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector \
  -e PCAM_API_URL=http://127.0.0.1:8765 -e PCAM_API_KEY=dev-key \
  uv run --directory "$PWD" server.py
```

The Inspector does not pass your shell environment to the server, so the variables go in with `-e`. Add `--cli --method tools/list` after `inspector` to run it without the browser UI.

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "pcam-inference": {
      "command": "uv",
      "args": ["run", "--directory", "/ABSOLUTE/PATH/TO/pcam-mcp-server", "server.py"],
      "env": {
        "PCAM_API_URL": "http://127.0.0.1:8765",
        "PCAM_API_KEY": "dev-key"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add pcam-inference \
  -e PCAM_API_URL=http://127.0.0.1:8765 -e PCAM_API_KEY=dev-key \
  -- uv run --directory "$PWD" server.py
```

## Error handling

Invalid input, timeouts, a rejected API key and API errors come back to the client as tool errors with a plain message, for example `File not found: /nope.png. Pass an absolute path to an image file.` Unexpected crashes return a generic message and are logged server-side.

## Adding a tool

1. Add the endpoint path constant at the top of `server.py`.
2. Write a function decorated with `@mcp.tool()`. Its docstring is what the assistant reads: say what it does, when to use it, when not to, and the exact input format.
3. Call the API through `_api(...)` so timeouts, auth and errors are handled the same way, and raise `ToolError` for failures the user can fix.
4. Include `DISCLAIMER` in the response.
5. Add tests in `test_server.py` for the happy path, invalid input and a timeout.
