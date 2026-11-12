"""Test doubles for the model provider (no API calls)."""

import json


class ScriptedProvider:
    """Returns prepared responses in order and records every request."""

    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(json.loads(json.dumps(request)))  # snapshot
        if not self.responses:
            raise AssertionError("The agent asked for more responses than scripted.")
        return self.responses.pop(0)


def reply(*blocks, stop="tool_use", tokens=(100, 20)):
    return {"content": list(blocks), "stop_reason": stop,
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}


def tool_use(name, args, id_="t1"):
    return {"type": "tool_use", "id": id_, "name": name, "input": args}
