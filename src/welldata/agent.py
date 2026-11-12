"""The tool-use loop shared by the LLM systems.

The model may call query tools any number of times, then must finish by calling
submit_answer. Tool errors go back to the model as error results with the fix in the
message, so it can retry. If the submitted text contains leftover tool-call markup, the
model gets one chance to resubmit. The loop stops after `max_steps` model calls.

Every tool call is recorded with its input and output (or error), so the grader can check
each claim against what the tools actually returned.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from welldata.answers import SUBMIT_ANSWER_TOOL, Answer, has_markup
from welldata.tools import ToolInputError

REMINDER = ("Please give your final answer by calling the submit_answer tool, with the numbers "
            "you state listed as claims.")
MARKUP_RETRY = ("The answer text contains tool-call markup (for example '</answer>' or "
                "'<parameter'). Call submit_answer again with the answer as plain text only.")


@dataclass
class AgentResult:
    answer: Answer | None
    tool_calls: list[str] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    steps: int = 0
    latency_s: float = 0.0
    error: str | None = None
    retries: int = 0


def _run_tool(tools: dict, name: str, args: dict) -> tuple[str, bool, object]:
    """Run one tool. Returns the text for the model, whether it is an error, and the output."""
    if name not in tools:
        return (f"There is no tool named {name!r}. Available tools: {', '.join(sorted(tools))}, "
                "submit_answer.", True, None)
    try:
        text = json.dumps(tools[name](**args), default=str)
    except ToolInputError as err:
        return str(err), True, None
    except TypeError as err:
        return (f"Wrong arguments for {name}: {err}. Use the argument names in the tool's "
                "input schema.", True, None)
    return text, False, json.loads(text)


def run_agent(provider, system: str, question: str, tools: dict, tool_specs: list | None = None,
              max_steps: int = 8, max_tokens: int = 4096) -> AgentResult:
    """Answer one question. `tools` maps tool names to callables; `tool_specs` are their schemas."""
    specs = list(tool_specs or []) + [SUBMIT_ANSWER_TOOL]
    messages: list[dict] = [{"role": "user", "content": question}]
    result = AgentResult(answer=None)
    reminded = False
    start = time.perf_counter()

    for step in range(1, max_steps + 1):
        response = provider.create(system=system, messages=messages, tools=specs, max_tokens=max_tokens)
        result.steps = step
        for k in ("input_tokens", "output_tokens"):
            result.usage[k] += response.get("usage", {}).get(k, 0)
        content = response["content"]
        messages.append({"role": "assistant", "content": content})

        uses = [b for b in content if b.get("type") == "tool_use"]
        submit = next((b for b in uses if b["name"] == "submit_answer"), None)
        if submit is not None:
            result.answer = Answer.from_tool_input(submit["input"])
            if not has_markup(result.answer.text) or result.retries:
                break
            result.retries += 1
        if not uses:
            if reminded:
                result.error = "The model answered in text and didn't call submit_answer."
                break
            messages.append({"role": "user", "content": REMINDER})
            reminded = True
            continue

        results = []
        for use in uses:
            if use is submit:
                results.append({"type": "tool_result", "tool_use_id": use["id"], "content": MARKUP_RETRY,
                                "is_error": True})
                continue
            args = use.get("input") or {}
            result.tool_calls.append(use["name"])
            text, is_error, output = _run_tool(tools, use["name"], args)
            result.tool_outputs.append({"name": use["name"], "input": args, "output": output,
                                        "error": text if is_error else None})
            results.append({"type": "tool_result", "tool_use_id": use["id"], "content": text,
                            "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        if result.answer is None:
            result.error = f"No answer after {max_steps} steps."

    result.latency_s = time.perf_counter() - start
    return result
