import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('provider_traces', ROOT / 'experiments/stark/harness_fixes/pi_context.py')
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


class ProviderTraceTests(unittest.TestCase):
    def normalize(self, payload):
        self.assertTrue(callable(getattr(context, 'provider_items', None)), 'Provider-aware request normalization is missing')
        return context.provider_items(payload)

    def test_responses_tool_calls_and_outputs_are_preserved(self):
        items = [{'type': 'function_call', 'call_id': 'call_1', 'name': 'bash'},
                 {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'synthetic output'}]
        self.assertEqual(self.normalize({'input': items}), items)

    def test_messages_tool_use_and_text_results_are_normalized(self):
        payload = {'system': 'synthetic system', 'messages': [
            {'role': 'user', 'content': 'synthetic question'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call_1', 'name': 'bash', 'input': {'command': 'true'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'}]}]}]}
        self.assertEqual(self.normalize(payload), [
            {'type': 'function_call', 'call_id': 'call_1', 'name': 'bash'},
            {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'first\nsecond'}])
        self.assertEqual(payload['system'], 'synthetic system')

    def test_messages_string_tool_result_is_preserved(self):
        self.assertEqual(self.normalize({'messages': [{'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'call_2', 'content': 'synthetic error', 'is_error': True}]}]})[0]['output'], 'synthetic error')

    def test_unknown_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            self.normalize({'unexpected': []})

    def test_messages_context_inspection_checks_tool_payload(self):
        event = {'messages': [{'role': 'toolResult', 'toolCallId': 'call_3', 'toolName': 'bash', 'content': [{'type': 'text', 'text': 'synthetic'}]}],
                 'payload': {'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_3', 'content': 'synthetic'}]}]}}
        self.assertEqual(context.inspect(event)[0]['text_chars'], 9)
        event['payload']['messages'][0]['content'][0]['content'] = 'different'
        with self.assertRaises(AssertionError):
            context.inspect(event)


if __name__ == '__main__':
    unittest.main()
