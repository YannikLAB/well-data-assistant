"""The provider layer: how the agent talks to a model.

A provider has one method, `create(system=..., messages=..., tools=..., max_tokens=...)`,
and returns a plain dict with `content` (a list of blocks), `stop_reason` and `usage`.
Messages and blocks use the Anthropic Messages API format; another provider would
translate to and from it. `CachedProvider` stores every response on disk so repeated
runs are free and repeatable.

Current Claude models reject non-default temperature, so no sampling settings are sent;
run-to-run variation is measured by repeating runs instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# USD per million tokens (input, output), from platform.claude.com/docs pricing, Nov 2026.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),   # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def estimate_cost(model: str, usage: dict) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    cost = (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6
    return round(cost, 6)


class AnthropicProvider:
    def __init__(self, model: str, client=None):
        self.model = model
        if client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add a line ANTHROPIC_API_KEY=<your key> to .env "
                    "in the project folder (see .env.example), then run the command again.")
            import anthropic  # reads ANTHROPIC_API_KEY from the environment

            client = anthropic.Anthropic()
        self.client = client

    def create(self, *, system: str, messages: list, tools: list, max_tokens: int) -> dict:
        response = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages, tools=tools)
        # Keep every block (including thinking blocks and their signatures) so the agent can
        # pass assistant turns back unchanged, as the API requires during tool use.
        return response.model_dump(mode="json", exclude_none=True)


class CachedProvider:
    """Wraps a provider and stores each response under a hash of the full request."""

    def __init__(self, inner, cache_dir: Path, run_label: str = ""):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label
        self.hits = 0
        self.misses = 0

    def create(self, **request) -> dict:
        key = hashlib.sha256(json.dumps(
            {"model": self.model, "run": self.run_label, **request}, sort_keys=True, default=str
        ).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        self.misses += 1
        response = self.inner.create(**request)
        path.write_text(json.dumps(response))
        return response
