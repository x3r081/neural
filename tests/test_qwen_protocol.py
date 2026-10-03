from qwen_neural.generation import normalize_messages, split_response
import pytest


def test_tool_arguments_normalize_without_mutating_history():
    original = [{"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]}]
    copied = normalize_messages(original)
    assert copied[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert isinstance(original[0]["tool_calls"][0]["function"]["arguments"], str)


def test_reasoning_and_complete_tool_call():
    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "properties": {"path": {"type": "string"}, "count": {"type": "integer"}}}}}]
    raw = 'Inspect first.</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\n123\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>'
    reasoning, content, calls = split_response(raw, True, tools)
    assert reasoning == 'Inspect first.' and content == ''
    assert calls[0]['function'] == {'name': 'write', 'arguments': '{"path": "123", "count": 2}'}


def test_incomplete_thinking_and_tool_markup_are_not_answer_text():
    assert split_response('Still thinking', True) == ('Still thinking', '', [])
    tools = [{'type': 'function', 'function': {'name': 'x'}}]
    assert split_response('Hello<tool_cal', False, tools) == ('', 'Hello', [])
    assert split_response('Hello<tool_call>\n<function=x>', False, tools) == ('', 'Hello', [])
    assert split_response('Explain <tool_call>', False) == ('', 'Explain <tool_call>', [])


@pytest.mark.parametrize('messages', [[None], [{'role': 'user', 'content': [None]}],
    [{'role': 'assistant', 'content': None, 'tool_calls': [None]}]])
def test_malformed_messages_are_client_errors(messages):
    with pytest.raises(ValueError):
        normalize_messages(messages)


def test_unknown_and_unfinished_tools_are_explicit_errors():
    tools = [{'type': 'function', 'function': {'name': 'x'}}]
    with pytest.raises(ValueError, match='unavailable tool'):
        split_response('<tool_call><function=unknown></function></tool_call>', False, tools)
    with pytest.raises(ValueError, match='incomplete tool'):
        split_response('<tool_call><function=x>', False, tools, final=True)


def test_file_tool_preserves_content_newlines_inside_template_framing():
    import json
    tools = [{'type': 'function', 'function': {'name': 'write', 'parameters': {
        'properties': {'content': {'type': 'string'}}}}}]
    raw = '<tool_call><function=write><parameter=content>\n\nhello\n\n</parameter></function></tool_call>'
    _, _, calls = split_response(raw, False, tools, final=True)
    assert json.loads(calls[0]['function']['arguments'])['content'] == '\nhello\n'
