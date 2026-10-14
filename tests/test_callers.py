"""Offline caller regression tests; no agent process or model request."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names], type_ignores=[])
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace

class CallerTests(unittest.TestCase):
    def test_explicit_model_overrides_paper_preset(self):
        ns = functions('experiments/stark/dci_run/core.py', {'model_name'}, {'os': os})
        with patch.dict(os.environ, {'MODEL_NAME': 'claude-test', 'DCI_MODEL': 'gpt-6-astra'}, clear=True):
            self.assertEqual(ns['model_name'](), 'claude-test')

    def test_unknown_model_has_no_paper_price(self):
        ns = functions('experiments/stark/prompt_val/runner.py', {'cost'}, {'os': os})
        with patch.dict(os.environ, {'MODEL_PROVIDER': 'anthropic', 'MODEL_NAME': 'claude-test'}, clear=True):
            self.assertIsNone(ns['cost']({'usage_complete': True, 'usage_raw': {'input': 100, 'output': 20, 'cacheRead': 0, 'cacheWrite': 0}}, 'pi'))

    def test_guard_blocks_cached_results_from_different_model(self):
        import json
        ns = functions('experiments/stark/prompt_val/runner.py', {'guard_run_configuration'}, {'Path': Path, 'os': os, 'base': SimpleNamespace(read_json=lambda path: json.loads(path.read_text()), save=lambda path, data: path.write_text(json.dumps(data)))})
        self.assertTrue(callable(ns.get('guard_run_configuration')), 'Run configuration guard missing')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol.json'
            ns['guard_run_configuration'](path, {'pi': {'provider': 'openai', 'model': 'first', 'api': 'openai-responses'}})
            with self.assertRaises(ValueError):
                ns['guard_run_configuration'](path, {'pi': {'provider': 'anthropic', 'model': 'second', 'api': 'anthropic-messages'}})

    def test_guard_blocks_unlabelled_legacy_results_for_new_model(self):
        import json
        ns = functions('experiments/stark/prompt_val/runner.py', {'guard_run_configuration'},
                       {'Path': Path, 'os': os, 'base': SimpleNamespace(read_json=lambda path: json.loads(path.read_text()), save=lambda path, data: path.write_text(json.dumps(data)))})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'result.json').write_text('{"terminal": true}')
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                ns['guard_run_configuration'](root / 'protocol.json', {'pi': {'provider': 'openai', 'model': 'gpt-5.4', 'api': 'openai-responses'}})

    def test_anthropic_harness_preserves_selected_model(self):
        import importlib.util
        import json
        from unittest.mock import Mock
        spec = importlib.util.spec_from_file_location('provider_test_callers', ROOT / 'experiments/agent_base/provider.py')
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / 'pi-mono/packages/coding-agent/dist/cli.js'
            cli.parent.mkdir(parents=True)
            cli.write_text('')
            attempt = root / 'attempt'
            corpus = root / 'corpus'
            corpus.mkdir()
            config = provider.configure_protocol({'pi_max_turns': 300}, 'pi', {'MODEL_PROVIDER': 'anthropic', 'MODEL_NAME': 'claude-test'})
            def save(path, data):
                path.write_text(json.dumps(data))
            def build(attempt, row, arm, protocol):
                (attempt / 'agent').mkdir()
                save(attempt / 'agent/models.json', provider.pi_provider_config(protocol))
                return ['synthetic-agent'], {}
            base = SimpleNamespace(REPO=root, configure_protocol=provider.configure_protocol, build_command=build,
                                   preflight=lambda attempt, row, arm, env: save(attempt / 'preflight.json', {}),
                                   isolation=lambda *args: [], read_json=lambda path: json.loads(path.read_text()), save=save)
            stark = SimpleNamespace(ROOT=root, prepare=SimpleNamespace(SOURCE=root))
            ns = functions('experiments/stark/prompt_val/harness.py', {'prepare_attempt'},
                           {'Path': Path, 'base': base, 'ARMS': {'pi': {'harness': 'pi'}}, 'stark': stark,
                            'subprocess': SimpleNamespace(run=Mock())})
            with patch.dict(os.environ, {'MODEL_PROVIDER': 'anthropic', 'MODEL_NAME': 'claude-test'}, clear=True):
                ns['prepare_attempt'](attempt, {'split': 'val', 'corpus_dir': str(corpus)}, config, 'pi', 'synthetic question')
            models = json.loads((attempt / 'agent/models.json').read_text())
            self.assertEqual(models['providers']['anthropic']['models'][0]['id'], 'claude-test')
            self.assertEqual(models['providers']['anthropic']['apiKey'], 'ANTHROPIC_API_KEY')

if __name__ == '__main__':
    unittest.main()
