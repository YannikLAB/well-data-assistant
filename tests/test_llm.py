"""The Anthropic provider and the response cache, tested without network access."""

from types import SimpleNamespace

from welldata.llm import AnthropicProvider, CachedProvider, estimate_cost


class CountingProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        return {"content": [{"type": "text", "text": f"call {self.calls}"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}}


REQUEST = {"system": "s", "messages": [{"role": "user", "content": "q"}], "tools": [], "max_tokens": 100}


def test_cache_returns_the_stored_response_for_the_same_request(tmp_path):
    inner = CountingProvider()
    cached = CachedProvider(inner, tmp_path)
    first = cached.create(**REQUEST)
    second = cached.create(**REQUEST)
    assert first == second and inner.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_run_label_forces_a_fresh_call(tmp_path):
    inner = CountingProvider()
    CachedProvider(inner, tmp_path, run_label="run1").create(**REQUEST)
    CachedProvider(inner, tmp_path, run_label="run2").create(**REQUEST)
    assert inner.calls == 2


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        response = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2}}
        return SimpleNamespace(model_dump=lambda **kw: response)


def test_anthropic_provider_sends_the_request_without_sampling_settings():
    messages = FakeMessages()
    provider = AnthropicProvider("claude-haiku-5-5", client=SimpleNamespace(messages=messages))
    result = provider.create(**REQUEST)
    assert messages.kwargs["model"] == "claude-haiku-5-5"
    assert messages.kwargs["max_tokens"] == 100 and messages.kwargs["system"] == "s"
    assert "temperature" not in messages.kwargs  # current models reject non-default temperature
    assert result["content"][0]["text"] == "hi"


def test_cost_estimate_uses_the_model_price():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert estimate_cost("claude-haiku-5-5", usage) == 0.60
    assert estimate_cost("unknown-model", usage) is None
