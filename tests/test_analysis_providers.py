"""Run the real offline analysis readers on native provider request fixtures."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments/stark/analysis_final"
sys.path.insert(0, str(ANALYSIS))
import common
import coevolution

spec = importlib.util.spec_from_file_location("analysis_case_trace", ANALYSIS / "cases/trace.py")
case_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_trace)


class AnalysisProviderTests(unittest.TestCase):
    def test_new_models_do_not_inherit_historical_prices(self):
        usage = {"input_tokens": 100, "output_tokens": 10}
        for provider, model in [("openai", "gpt-5.4"), ("anthropic", "claude-sonnet-4-6")]:
            self.assertIsNone(common.price_equivalent({"provider": provider, "model": model, "usage_raw": usage}))
        self.assertIsNotNone(common.price_equivalent({"usage_raw": usage}))

    def check_readers(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / "artifacts").mkdir()
            (attempt / "artifacts/events.jsonl").write_text(json.dumps({
                "type": "provider_request_context", "payload": payload}) + "\n")
            calls = common.pi_calls(attempt, union=False)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], {"command": "printf synthetic"})
            self.assertEqual(calls[0]["output"], "synthetic output")
            self.assertTrue(calls[0]["visible"])
            trajectory = coevolution.pi_trace(attempt, union=False)
            self.assertEqual(trajectory["prompt"], "synthetic question")
            self.assertEqual(trajectory["calls"][0]["cmd"], "printf synthetic")
            self.assertEqual(trajectory["calls"][0]["out"], "synthetic output")
            rows, _ = case_trace.pi_trace(attempt)
            self.assertEqual(rows[0]["args"], {"command": "printf synthetic"})
            self.assertEqual(rows[0]["output"], "synthetic output")

    def test_anthropic_messages_reach_all_analysis_readers(self):
        self.check_readers({"system": [{"type": "text", "text": "synthetic system"}], "messages": [
            {"role": "user", "content": [{"type": "text", "text": "synthetic question"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "bash",
                                               "input": {"command": "printf synthetic"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                          "content": [{"type": "text", "text": "synthetic output"}]}]}]})

    def test_openai_responses_analysis_is_unchanged(self):
        self.check_readers({"input": [
            {"role": "user", "content": "synthetic question"},
            {"type": "function_call", "call_id": "call_1", "name": "bash",
             "arguments": json.dumps({"command": "printf synthetic"})},
            {"type": "function_call_output", "call_id": "call_1", "output": "synthetic output"}]})


if __name__ == "__main__":
    unittest.main()
