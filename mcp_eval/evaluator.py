"""Evaluation runner for mcp-eval.

Loads a test dataset, runs each question through an Agent backed by MCP tools,
judges the answers with an LLM, and writes structured results to JSON files.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from .agent import Agent, AgentResult
from .eval_config import EvalConfig
from .llm import LLMClient
from .schema import LLMProvider, Message
from .tools.mcp_loader import cleanup_mcp_connections, get_connected_server_names, load_mcp_tools_async
from .utils.terminal_utils import calculate_display_width, truncate_with_ellipsis


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


BOX_WIDTH = 58

_CONFIG_DIR = Path(__file__).parent / "config"


JUDGE_PROMPT_TEMPLATE = """Given the question: {question}

Judge whether the predicted answer is correct. Matching key information is sufficient:

Predicted: {prediction}
Ground truth: {ground_truth}
{tool_calls_section}
Return only True or False."""

JUDGE_TOOL_CALLS_TEMPLATE = """
The agent called the following tools during execution, with MCP server responses:
{tool_calls_text}
"""

JUDGE_TOOL_CALL_ITEM = "Tool: {name}\nArguments: {arguments}\nResult: {result}"


@dataclass
class TestCase:
    """A single test question from the dataset."""

    unique_id: int
    prompt: str
    answer: str


@dataclass
class QuestionResult:
    """Evaluation result for a single test question."""

    unique_id: int
    prompt: str
    ground_truth: str
    prediction: str
    passed: bool
    judge_raw: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    steps: int = 0
    timed_out: bool = False
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation report across all questions."""

    timestamp: str
    agent_llm: str
    judge_llm: str
    mcp_servers: list[str]
    dataset: str = ""
    tool_desc_length: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_elapsed_seconds: float = 0.0
    avg_elapsed_seconds: float = 0.0
    total_steps: int = 0
    avg_steps: float = 0.0
    total_questions: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    results: list[QuestionResult] = field(default_factory=list)


def load_dataset(path: str) -> list[TestCase]:
    """Load test cases from a JSON or JSONL file.

    JSON format: an array of objects with unique_id, Prompt, and Answer fields.
    JSONL format: each line is a JSON object with the same fields.

    Args:
        path: Path to the dataset file.

    Returns:
        List of TestCase objects.
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            items = json.load(f)
        else:
            items = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

    for item in items:
        cases.append(TestCase(
            unique_id=item["unique_id"],
            prompt=item["Prompt"],
            answer=item["Answer"],
            ))
    return cases


def _compute_tool_desc_length(tools) -> int:
    """Calculate total character length of all tool names + descriptions."""
    total = 0
    for tool in tools:
        total += len(tool.name) + len(tool.description)
    return total


async def judge_answer(
    api_key: str,
    api_base: str,
    model: str,
    question: str,
    prediction: str,
    ground_truth: str,
    tool_calls: list[dict] | None = None,
) -> tuple[bool, str]:
    """Use an LLM as judge to evaluate whether prediction matches ground truth.

    Uses the Anthropic protocol directly with temperature=0.01 for deterministic
    judging. Returns True if the judge response contains "true".

    Args:
        api_key: API key for the judge LLM.
        api_base: API base URL (Anthropic protocol).
        model: Model name for the judge.
        question: The original question.
        prediction: The agent's answer.
        ground_truth: The expected answer.
        tool_calls: Optional list of tool call records with name, arguments, result.

    Returns:
        Tuple of (passed: bool, judge_raw_response: str).
    """
    client = anthropic.AsyncAnthropic(
        base_url=api_base,
        api_key=api_key,
        default_headers={"Authorization": f"Bearer {api_key}"},
    )

    tool_calls_section = ""
    if tool_calls:
        items = []
        for tc in tool_calls:
            items.append(JUDGE_TOOL_CALL_ITEM.format(
                name=tc.get("name", ""),
                arguments=tc.get("arguments", ""),
                result=tc.get("result", ""),
            ))
        tool_calls_section = JUDGE_TOOL_CALLS_TEMPLATE.format(
            tool_calls_text="\n".join(items),
        )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        prediction=prediction,
        ground_truth=ground_truth,
        tool_calls_section=tool_calls_section,
    )
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
    )
    text = response.content[0].text
    passed = "true" in text.lower()
    return passed, text


def _serialize_messages(messages: list[Message]) -> list[dict]:
    """Convert Message objects to JSON-serializable dicts."""
    result = []
    for msg in messages:
        d: dict = {"role": msg.role}
        if isinstance(msg.content, str):
            d["content"] = msg.content
        else:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        result.append(d)
    return result


def _print_progress(case: TestCase, idx: int, total: int, result: QuestionResult) -> None:
    """Print a single question's evaluation result to the terminal."""
    if result.passed:
        status = f"{Colors.BRIGHT_GREEN}PASS{Colors.RESET}"
    else:
        status = f"{Colors.BRIGHT_RED}FAIL{Colors.RESET}"
    timeout_flag = f" {Colors.BRIGHT_YELLOW}(TIMEOUT){Colors.RESET}" if result.timed_out else ""
    print(f"[{idx}/{total}] Q{case.unique_id}: {status}{timeout_flag}")
    print(f"  Time: {result.elapsed_seconds:.1f}s | Tokens: {result.input_tokens}+{result.output_tokens} | Steps: {result.steps}")
    print()


def _print_banner(report: EvalReport) -> None:
    """Print evaluation report header."""
    title = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}🤖 Evaluation Report{Colors.RESET}"
    title_width = calculate_display_width(title)
    side_width = (BOX_WIDTH - title_width) // 2
    print()
    print(f"{Colors.DIM}{'─' * side_width}{Colors.RESET} {title} {Colors.DIM}{'─' * side_width}{Colors.RESET}")


def _print_box_line(text: str) -> None:
    """Print a single line inside a thin box, padded to BOX_WIDTH."""
    padding = max(0, BOX_WIDTH + 1 - calculate_display_width(text))
    print(f"{Colors.DIM}│{Colors.RESET}{text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")


def _rate_color(rate: float) -> str:
    """Return color code based on pass rate."""
    if rate >= 0.8:
        return Colors.BRIGHT_GREEN
    elif rate >= 0.5:
        return Colors.BRIGHT_YELLOW
    return Colors.BRIGHT_RED


def print_report(report: EvalReport) -> None:
    """Print the final evaluation summary to the terminal with box drawing."""
    _print_banner(report)

    print(f"{Colors.DIM}╭{'─' * BOX_WIDTH}╮{Colors.RESET}")

    _print_box_line(f" {Colors.BRIGHT_CYAN}Agent LLM:{Colors.RESET}    {report.agent_llm}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Judge LLM:{Colors.RESET}    {report.judge_llm}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}MCP Servers:{Colors.RESET}  {' + '.join(report.mcp_servers)}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Dataset:{Colors.RESET}      {report.dataset}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Tool Desc:{Colors.RESET}    {report.tool_desc_length}")

    print(f"{Colors.DIM}├{'─' * BOX_WIDTH}┤{Colors.RESET}")

    rate_color = _rate_color(report.pass_rate)
    _print_box_line(f" {Colors.BOLD}Total:{Colors.RESET}  {report.total_questions}")
    _print_box_line(f" {Colors.BOLD}Passed:{Colors.RESET} {report.passed}")
    _print_box_line(f" {Colors.BOLD}Rate:{Colors.RESET}   {rate_color}{report.pass_rate:.1%}{Colors.RESET}")

    print(f"{Colors.DIM}├{'─' * BOX_WIDTH}┤{Colors.RESET}")

    _print_box_line(f" Total Input Tokens:      {report.total_input_tokens}")
    _print_box_line(f" Total Output Tokens:     {report.total_output_tokens}")
    _print_box_line(f" Avg Input Tokens:        {report.avg_input_tokens:.0f}")
    _print_box_line(f" Avg Output Tokens:       {report.avg_output_tokens:.0f}")
    _print_box_line(f" Total Steps:             {report.total_steps}")
    _print_box_line(f" Avg Steps:               {report.avg_steps:.1f}")
    _print_box_line(f" Total Time:              {report.total_elapsed_seconds:.1f}s")
    _print_box_line(f" Avg Time:                {report.avg_elapsed_seconds:.1f}s")

    print(f"{Colors.DIM}╰{'─' * BOX_WIDTH}╯{Colors.RESET}")


def save_results(report: EvalReport, output_dir: str = "results") -> None:
    """Write evaluation results to JSON files.

    Creates a folder named {server1_server2}_{timestamp} containing:
    - summary.json: Aggregate metrics only (no per-question data).
    - details.json: All per-question results as a single JSON array.

    Args:
        report: The evaluation report to save.
        output_dir: Directory to write results into.
    """
    folder_name = f"{'_'.join(report.mcp_servers)}_{report.timestamp}"
    result_dir = Path(output_dir) / folder_name
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": report.dataset,
        "tool_desc_length": report.tool_desc_length,
        "total_input_tokens": report.total_input_tokens,
        "total_output_tokens": report.total_output_tokens,
        "avg_input_tokens": report.avg_input_tokens,
        "avg_output_tokens": report.avg_output_tokens,
        "total_elapsed_seconds": report.total_elapsed_seconds,
        "avg_elapsed_seconds": report.avg_elapsed_seconds,
        "total_steps": report.total_steps,
        "avg_steps": report.avg_steps,
        "pass_rate": report.pass_rate,
        "total_questions": report.total_questions,
        "passed": report.passed,
        "timestamp": report.timestamp,
        "agent_llm": report.agent_llm,
        "judge_llm": report.judge_llm,
        "mcp_servers": report.mcp_servers,
    }
    summary_path = result_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")

    details = [asdict(r) for r in report.results]
    details_path = result_dir / "details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Details saved to: {details_path}")


async def run_eval(
    config_path: str,
    mcp_config_path: str,
    dataset_path: str,
) -> EvalReport:
    """Main evaluation entry point.

    Loads eval config, connects to all MCP servers from the specified MCP config,
    runs each test question through an Agent, judges answers, and returns a
    complete EvalReport.

    Args:
        config_path: Path to the evaluation YAML configuration file.
        mcp_config_path: Path to the MCP server configuration JSON file.
        dataset_path: Path to the dataset file (JSON or JSONL).

    Returns:
        EvalReport with aggregated and per-question results.
    """
    config = EvalConfig.from_yaml(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    dataset_name = Path(dataset_path).stem
    test_cases = load_dataset(dataset_path)

    _print_eval_start_banner(
        total=len(test_cases),
        agent_model=config.agent_llm.model,
        judge_model=config.judge_llm.model,
        dataset=dataset_name,
        timeout=config.timeout,
    )

    # Connect MCP tools (all servers in config)
    if not Path(mcp_config_path).exists():
        raise FileNotFoundError(f"MCP config not found: {mcp_config_path}")
    print(f"Connecting to MCP servers from {mcp_config_path}...")
    mcp_tools = await load_mcp_tools_async(mcp_config_path)
    if not mcp_tools:
        raise RuntimeError("No MCP tools loaded — cannot run evaluation")
    tool_desc_length = _compute_tool_desc_length(mcp_tools)

    # Create agent LLM client
    agent_llm = LLMClient(
        api_key=config.agent_llm.api_key,
        provider=LLMProvider.ANTHROPIC if config.agent_llm.provider == "anthropic" else LLMProvider.OPENAI,
        api_base=config.agent_llm.api_base,
        model=config.agent_llm.model,
    )

    # Resolve MCP server names from actually connected servers
    mcp_server_names = get_connected_server_names()

    # Run each test case
    results: list[QuestionResult] = []
    total = len(test_cases)

    for idx, case in enumerate(test_cases, 1):
        system_prompt = config.system_prompt
        if not system_prompt:
            raise ValueError("system_prompt is required in the evaluation config")
        agent = Agent(
            llm_client=agent_llm,
            system_prompt=system_prompt,
            tools=mcp_tools,
            max_steps=config.max_steps,
            silent=True,
        )
        agent.add_user_message(case.prompt)

        agent_result: Optional[AgentResult] = None
        timed_out = False

        try:
            agent_result = await asyncio.wait_for(
                agent.run(),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True

        # Build QuestionResult
        if timed_out:
            qr = QuestionResult(
                unique_id=case.unique_id,
                prompt=case.prompt,
                ground_truth=case.answer,
                prediction="[TIMEOUT]",
                passed=False,
                judge_raw="",
                timed_out=True,
                elapsed_seconds=config.timeout,
                messages=_serialize_messages(agent.messages),
                tool_calls=[],
            )
        elif agent_result is not None:
            qr = QuestionResult(
                unique_id=case.unique_id,
                prompt=case.prompt,
                ground_truth=case.answer,
                prediction=agent_result.answer,
                passed=False,
                judge_raw="",
                input_tokens=agent_result.input_tokens,
                output_tokens=agent_result.output_tokens,
                elapsed_seconds=agent_result.elapsed_seconds,
                steps=agent_result.steps,
                timed_out=False,
                messages=_serialize_messages(agent_result.messages),
                tool_calls=agent_result.tool_calls,
            )
        else:
            qr = QuestionResult(
                unique_id=case.unique_id,
                prompt=case.prompt,
                ground_truth=case.answer,
                prediction="[ERROR]",
                passed=False,
                judge_raw="",
                messages=[],
                tool_calls=[],
            )

        # Judge (skip for timeout/error)
        if not qr.timed_out and qr.prediction not in ("[TIMEOUT]", "[ERROR]"):
            try:
                passed, judge_raw = await judge_answer(
                    api_key=config.judge_llm.api_key,
                    api_base=config.judge_llm.api_base,
                    model=config.judge_llm.model,
                    question=case.prompt,
                    prediction=qr.prediction,
                    ground_truth=case.answer,
                    tool_calls=qr.tool_calls,
                )
                qr.passed = passed
                qr.judge_raw = judge_raw
            except Exception as e:
                qr.judge_raw = f"Judge error: {e}"

        results.append(qr)
        _print_progress(case, idx, total, qr)

    # Cleanup MCP connections (suppress anyio cancel scope errors on shutdown)
    try:
        await cleanup_mcp_connections()
    except Exception:
        pass

    # Aggregate
    passed_count = sum(1 for r in results if r.passed)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_time = sum(r.elapsed_seconds for r in results)
    total_steps = sum(r.steps for r in results)

    report = EvalReport(
        timestamp=timestamp,
        agent_llm=config.agent_llm.model,
        judge_llm=config.judge_llm.model,
        mcp_servers=mcp_server_names,
        dataset=dataset_name,
        tool_desc_length=tool_desc_length,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        avg_input_tokens=total_input / total if total > 0 else 0.0,
        avg_output_tokens=total_output / total if total > 0 else 0.0,
        total_elapsed_seconds=total_time,
        avg_elapsed_seconds=total_time / total if total > 0 else 0.0,
        total_steps=total_steps,
        avg_steps=total_steps / total if total > 0 else 0.0,
        total_questions=total,
        passed=passed_count,
        pass_rate=passed_count / total if total > 0 else 0.0,
        results=results,
    )

    print_report(report)
    save_results(report)

    return report


def _print_eval_start_banner(total: int, agent_model: str, judge_model: str, dataset: str, timeout: int) -> None:
    """Print the evaluation start banner with config info."""
    banner_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}🤖 mcp-eval{Colors.RESET}"
    banner_width = calculate_display_width(banner_text)
    total_padding = BOX_WIDTH - banner_width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{'═' * BOX_WIDTH}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}{' ' * left_padding}{banner_text}{' ' * right_padding}{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{'═' * BOX_WIDTH}╝{Colors.RESET}")

    print(f"{Colors.DIM}╭{'─' * BOX_WIDTH}╮{Colors.RESET}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Agent LLM:{Colors.RESET}  {agent_model}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Judge LLM:{Colors.RESET}  {judge_model}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Dataset:{Colors.RESET}    {dataset}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Questions:{Colors.RESET}  {total}")
    _print_box_line(f" {Colors.BRIGHT_CYAN}Timeout:{Colors.RESET}    {timeout}s")
    print(f"{Colors.DIM}╰{'─' * BOX_WIDTH}╯{Colors.RESET}")
    print()
