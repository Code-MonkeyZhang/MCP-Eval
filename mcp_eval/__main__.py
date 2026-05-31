"""Entry point for python -m mcp_eval.

Supports two modes:
  python -m mcp_eval eval --mcp <mcp.json> --config <eval.yaml> --dataset <data.json>
  python -m mcp_eval login --mcp <mcp.json> [options]
"""

import asyncio
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _parse_eval_args(argv: list[str]) -> dict | None:
    """Parse eval subcommand arguments.

    Returns a dict with parsed options if eval mode is requested, None otherwise.

    All three options are required:
      --mcp PATH       MCP server config JSON
      --config PATH    Evaluation config YAML
      --dataset PATH   Dataset file (JSON or JSONL)
    """
    if len(argv) < 2 or argv[1] != "eval":
        return None

    opts = {
        "mcp": None,
        "config": None,
        "dataset": None,
    }

    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg in ("--mcp", "--config", "--dataset") and i + 1 < len(argv):
            key = arg.lstrip("-")
            opts[key] = argv[i + 1]
            i += 2
        elif arg.startswith("--mcp="):
            opts["mcp"] = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--config="):
            opts["config"] = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--dataset="):
            opts["dataset"] = arg.split("=", 1)[1]
            i += 1
        else:
            i += 1

    missing = [k for k, v in opts.items() if v is None]
    if missing:
        flags = ", ".join(f"--{k}" for k in missing)
        print(f"Error: missing required options: {flags}")
        print("Usage: python -m mcp_eval eval --mcp <mcp.json> --config <eval.yaml> --dataset <data.json>")
        return None

    return opts


def _parse_login_args(argv: list[str]) -> dict | None:
    """Parse login subcommand arguments.

    Returns a dict with parsed options if login mode is requested, None otherwise.

    Supported options:
      --mcp PATH       MCP server config JSON (required)
      --server NAME    Which MCP server to log in to
    """
    if len(argv) < 2 or argv[1] != "login":
        return None

    opts = {
        "mcp": None,
        "server": None,
    }

    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg in ("--mcp", "--server") and i + 1 < len(argv):
            key = arg.lstrip("-")
            opts[key] = argv[i + 1]
            i += 2
        elif arg.startswith("--mcp="):
            opts["mcp"] = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--server="):
            opts["server"] = arg.split("=", 1)[1]
            i += 1
        else:
            i += 1

    if opts["mcp"] is None:
        print("Error: --mcp <mcp_config_path> is required for login mode")
        print("Usage: python -m mcp_eval login --mcp <mcp.json> [--server <name>]")
        return None

    return opts


def _run_async(coro):
    """Run an async coroutine with suppressed shutdown warnings."""
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def main():
    """Route to the appropriate mode based on command-line arguments."""
    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        opts = _parse_eval_args(sys.argv)
        if opts is not None:
            from mcp_eval.evaluator import run_eval
            _run_async(run_eval(
                config_path=opts["config"],
                mcp_config_path=opts["mcp"],
                dataset_path=opts["dataset"],
            ))
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "login":
        opts = _parse_login_args(sys.argv)
        if opts is not None:
            from mcp_eval.tools.mcp_loader import login_mcp_server
            _run_async(login_mcp_server(
                config_path=opts["mcp"],
                server_name=opts["server"],
            ))
        return

    print("Usage: python -m mcp_eval <eval|login> [options]")
    print("  eval  --mcp <mcp.json> --config <eval.yaml> --dataset <data.json>")
    print("  login --mcp <mcp.json> [--server <name>]")


if __name__ == "__main__":
    main()
