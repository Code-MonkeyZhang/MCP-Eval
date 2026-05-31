"""Core Agent implementation."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import tiktoken

from .llm import LLMClient
from .schema import Message
from .tools.base import Tool, ToolResult
from .utils import calculate_display_width


@dataclass
class AgentResult:
    """Structured result returned by Agent.run() in silent mode.

    Attributes:
        answer: The final text response from the agent.
        input_tokens: Total input tokens consumed across all LLM calls.
        output_tokens: Total output tokens consumed across all LLM calls.
        elapsed_seconds: Wall-clock time of the entire run in seconds.
        steps: Number of agent steps executed.
        messages: Full conversation message history.
        tool_calls: Log of all tool invocations [{name, arguments, result}].
    """

    answer: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    steps: int = 0
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


# ANSI color codes
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,
        silent: bool = False,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.workspace_dir = Path(workspace_dir)
        self.silent = silent

        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        self.api_total_tokens: int = 0

    def _print(self, *args, **kwargs):
        """Print only when not in silent mode."""
        if not self.silent:
            print(*args, **kwargs)

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    async def run(self) -> AgentResult:
        """Execute agent loop until task is complete or max steps reached.

        Returns:
            AgentResult containing the final answer, token usage, timing, and conversation history.
        """
        step = 0
        run_start_time = perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_calls_log: list[dict] = []

        while step < self.max_steps:
            step_start_time = perf_counter()

            BOX_WIDTH = 58
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}💭 Step {step + 1}/{self.max_steps}{Colors.RESET}"
            step_display_width = calculate_display_width(step_text)
            padding = max(0, BOX_WIDTH - 1 - step_display_width)

            self._print(f"\n{Colors.DIM}╭{'─' * BOX_WIDTH}╮{Colors.RESET}")
            self._print(f"{Colors.DIM}│{Colors.RESET} {step_text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")
            self._print(f"{Colors.DIM}╰{'─' * BOX_WIDTH}╯{Colors.RESET}")

            tool_list = list(self.tools.values())

            try:
                response = await self.llm.generate(messages=self.messages, tools=tool_list)
            except Exception as e:
                from .retry import RetryExhaustedError

                if isinstance(e, RetryExhaustedError):
                    error_msg = f"LLM call failed after {e.attempts} retries\nLast error: {str(e.last_exception)}"
                    self._print(f"\n{Colors.BRIGHT_RED}❌ Retry failed:{Colors.RESET} {error_msg}")
                else:
                    error_msg = f"LLM call failed: {str(e)}"
                    self._print(f"\n{Colors.BRIGHT_RED}❌ Error:{Colors.RESET} {error_msg}")
                return AgentResult(
                    answer=error_msg,
                    input_tokens=total_prompt_tokens,
                    output_tokens=total_completion_tokens,
                    elapsed_seconds=perf_counter() - run_start_time,
                    steps=step,
                    messages=self.messages.copy(),
                    tool_calls=tool_calls_log,
                )

            if response.usage:
                self.api_total_tokens = response.usage.total_tokens
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            if response.thinking:
                self._print(f"\n{Colors.BOLD}{Colors.MAGENTA}🧠 Thinking:{Colors.RESET}")
                self._print(f"{Colors.DIM}{response.thinking}{Colors.RESET}")

            if response.content:
                self._print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                self._print(f"{response.content}")

            if not response.tool_calls:
                step_elapsed = perf_counter() - step_start_time
                total_elapsed = perf_counter() - run_start_time
                self._print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")
                return AgentResult(
                    answer=response.content,
                    input_tokens=total_prompt_tokens,
                    output_tokens=total_completion_tokens,
                    elapsed_seconds=total_elapsed,
                    steps=step + 1,
                    messages=self.messages.copy(),
                    tool_calls=tool_calls_log,
                )

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

                self._print(f"\n{Colors.BRIGHT_YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{function_name}{Colors.RESET}")

                self._print(f"{Colors.DIM}   Arguments:{Colors.RESET}")
                truncated_args = {}
                for key, value in arguments.items():
                    value_str = str(value)
                    if len(value_str) > 200:
                        truncated_args[key] = value_str[:200] + "..."
                    else:
                        truncated_args[key] = value
                args_json = json.dumps(truncated_args, indent=2, ensure_ascii=False)
                for line in args_json.split("\n"):
                    self._print(f"   {Colors.DIM}{line}{Colors.RESET}")

                if function_name not in self.tools:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=f"Unknown tool: {function_name}",
                    )
                else:
                    try:
                        tool = self.tools[function_name]
                        result = await tool.execute(**arguments)
                    except Exception as e:
                        import traceback

                        error_detail = f"{type(e).__name__}: {str(e)}"
                        error_trace = traceback.format_exc()
                        result = ToolResult(
                            success=False,
                            content="",
                            error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
                        )

                tool_calls_log.append({
                    "name": function_name,
                    "arguments": arguments,
                    "result": result.content if result.success else f"Error: {result.error}",
                })

                if result.success:
                    result_text = result.content
                    if len(result_text) > 300:
                        result_text = result_text[:300] + f"{Colors.DIM}...{Colors.RESET}"
                    self._print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {result_text}")
                else:
                    self._print(f"{Colors.BRIGHT_RED}✗ Error:{Colors.RESET} {Colors.RED}{result.error}{Colors.RESET}")

                tool_msg = Message(
                    role="tool",
                    content=result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                self.messages.append(tool_msg)

            step_elapsed = perf_counter() - step_start_time
            total_elapsed = perf_counter() - run_start_time
            self._print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        self._print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        return AgentResult(
            answer=error_msg,
            input_tokens=total_prompt_tokens,
            output_tokens=total_completion_tokens,
            elapsed_seconds=perf_counter() - run_start_time,
            steps=step,
            messages=self.messages.copy(),
            tool_calls=tool_calls_log,
        )

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
