# mcp-eval
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An evaluation framework for [MCP Servers](https://modelcontextprotocol.io/). Given a test prompt set, this tool runs each prompt through a ReAct agent to evaluate your MCP's performance.

**🎯 Use cases:**

- Evaluate the tool-calling accuracy of your MCP Server
- Compare how different LLMs perform on the same MCP Server
- Regression test your MCP Server after updates or changes
- Benchmark multiple MCP Servers against the same test prompts

## How It Works

For each test question:

1. **Agent Execution** — An Agent is launched with your MCP Server's tools. The question is passed as a user prompt, and the Agent is free to call any available tools to complete the task.
2. **LLM-as-Judge** — A separate Judge LLM Agent compares the Agent's answer against the reference answer. Scoring is binary (pass/fail) — the reference is descriptive, allowing semantic flexibility.
3. **Report Generation** — Results are aggregated into structured JSON files under `results/`.

The Evaluation will generate a report for token usage, latency, agent loop consumption, and full tool call logs.

## 🚀 Quick Start

### Clone & Install

```bash
git clone https://github.com/Code-MonkeyZhang/mcp-eval.git
cd mcp-eval
uv sync
```

### Prepare a Dataset

Create a JSON file with test questions containing `unique_id`, `Prompt`, and `Answer` (a descriptive reference, not an exact match):

```json
[
  {
    "unique_id": 1,
    "Prompt": "List all my projects",
    "Answer": "Returned all projects with names and IDs"
  },
  {
    "unique_id": 2,
    "Prompt": "Show all uncompleted tasks in my inbox",
    "Answer": "Returned all uncompleted inbox tasks"
  }
]
```

### Create an Eval Config

```yaml
# eval.yaml
agent_llm:
  provider: anthropic   # "anthropic" or "openai"
  api_key: sk-xxx
  api_base: https://api.anthropic.com
  model: claude-sonnet-4-20250514

judge_llm:
  provider: anthropic
  api_key: sk-xxx
  api_base: https://api.anthropic.com
  model: claude-sonnet-4-20250514

system_prompt: |
  You are a helpful AI assistant with access to tools. Use the available tools to complete the user's task accurately.
max_steps: 50
timeout: 120
```

### Configure MCP Server

```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server"]
    }
  }
}
```

If your MCP Server requires OAuth:

```bash
python -m mcp_eval login --mcp mcp.json [--server <name>]
```

> [!NOTE]
> The `login` command will open a browser for OAuth authentication. After completing the login, the credentials will be saved locally for subsequent evaluation runs.

### Run Evaluation

```bash
python -m mcp_eval eval \
  --mcp mcp.json \
  --config eval.yaml \
  --dataset datasets/test.json
```

## 📋 Output

Results are saved to `results/<servers>_<timestamp>/`:

| File           | Description                                                                            |
| -------------- | -------------------------------------------------------------------------------------- |
| `summary.json` | Aggregate metrics: pass rate, token usage, latency, models used                        |
| `details.json` | Per-question breakdown: agent answer, judge decision, tool calls, full message history |

Example `summary.json`:

```json
{
  "total_questions": 2,
  "passed": 2,
  "pass_rate": 1.0,
  "avg_input_tokens": 4856,
  "avg_output_tokens": 126,
  "avg_elapsed_seconds": 11.8,
  "agent_llm": "glm-5.1",
  "judge_llm": "glm-5.1",
  "mcp_servers": ["my-server"]
}
```

## 📁 Project Structure

```
mcp-eval/
├── mcp_eval/
│   ├── evaluator.py         # Evaluation runner & report generation
│   ├── eval_config.py       # YAML config loader
│   ├── agent.py             # Core Agent with tool-calling loop
│   ├── __main__.py          # CLI entry point
│   ├── retry.py             # Retry logic
│   ├── llm/                 # LLM client (OpenAI & Anthropic)
│   ├── tools/               # MCP loader & tool implementations
│   ├── schema/              # Data models
│   └── utils/               # Terminal utilities
├── config/                  # Example configs
├── datasets/                # Test datasets (JSON/JSONL)
├── results/                 # Evaluation output (gitignored)
└── pyproject.toml           # Project metadata & deps
```

## License

[MIT](LICENSE)
