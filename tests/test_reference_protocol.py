import copy
import json
from types import SimpleNamespace

import pytest

from neural_reference.protocol import (ProtocolError, capabilities, render_chat_prompt,
                                    split_chat_output)


class TemplateTokenizer:
    def __init__(self, prompt_suffix=""):
        self.prompt_suffix = prompt_suffix
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return "rendered prompt" + self.prompt_suffix


def backend(model_type, prompt_suffix=""):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type),
                           tokenizer=TemplateTokenizer(prompt_suffix))


def tools():
    return [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"},
                                             "days": {"type": "integer"}},
        "required": ["city"], "additionalProperties": False}}}]


def test_qwen35_uses_template_and_xml_protocol_without_mutating_history():
    b = backend("qwen3_5_moe", "<think>")
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "c1", "type": "function", "function": {"name": "weather",
        "arguments": '{"city":"Oslo"}'}}]}, {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "user", "content": "weather?"}]
    original = copy.deepcopy(messages)
    prompt, thinking, schemas, protocol = render_chat_prompt(b, {
        "messages": messages, "tools": tools(),
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}})
    assert prompt.endswith("<think>") and thinking
    assert protocol == "qwen_xml" and "weather" in schemas
    assert messages == original
    templated_messages = b.tokenizer.calls[0][0]
    assert templated_messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Oslo"}

    raw = ('Think first.</think>\n\n<tool_call><function=weather>'
           '<parameter=city>\nOslo\n</parameter><parameter=days>2</parameter>'
           '</function></tool_call>')
    reason, content, calls = split_chat_output(raw, thinking=True, tools=tools(),
                                                tool_protocol=protocol, final=True)
    assert reason == "Think first." and content == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo", "days": 2}


def test_qwen3_json_tool_markup_and_plain_markup_behavior():
    protocol = capabilities(backend("qwen3_moe"))["tool_protocol"]
    assert protocol == "qwen_json"
    raw = 'literal before <tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>'
    reason, content, calls = split_chat_output(raw, thinking=False, tools=tools(),
                                                tool_protocol=protocol, final=True)
    assert reason == "" and content == "literal before "
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo"}
    _, literal, no_calls = split_chat_output(raw, thinking=False, tools=None,
                                             tool_protocol=protocol, final=True)
    assert literal == raw and no_calls == []


def test_mixtral_does_not_assume_tools_or_reasoning():
    b = backend("mixtral", "")
    assert capabilities(b) == {"tool_protocol": "none", "thinking": False,
                               "template_kwargs": False}
    with pytest.raises(ProtocolError, match="not supported"):
        render_chat_prompt(b, {"messages": [{"role": "user", "content": "hi"}], "tools": tools()})
    prompt, thinking, _, protocol = render_chat_prompt(b, {
        "messages": [{"role": "user", "content": "hi"}]})
    assert prompt == "rendered prompt" and not thinking and protocol == "none"
    raw = "<think>ordinary literal text</think>"
    assert split_chat_output(raw, thinking=False, tools=None, tool_protocol=protocol)[1] == raw


@pytest.mark.parametrize("arguments", ["[]", "{bad", {"city": 3}, {"city": "Oslo", "extra": 1}])
def test_output_tool_arguments_are_validated(arguments):
    payload = arguments if isinstance(arguments, str) else json.dumps(arguments)
    raw = f"<tool_call>{{\"name\":\"weather\",\"arguments\":{json.dumps(payload)}}}</tool_call>"
    with pytest.raises(ProtocolError):
        split_chat_output(raw, thinking=False, tools=tools(), tool_protocol="qwen_json", final=True)


def test_partial_qwen_tool_call_is_withheld_until_complete():
    _, content, calls = split_chat_output("visible<tool_call>{\"name\":", thinking=False,
        tools=tools(), tool_protocol="qwen_json", final=False)
    assert content == "visible" and calls == []
    with pytest.raises(ProtocolError, match="incomplete"):
        split_chat_output("visible<tool_call>{\"name\":", thinking=False,
            tools=tools(), tool_protocol="qwen_json", final=True)
