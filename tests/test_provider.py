import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AGENT_BASE = ROOT / 'experiments/agent_base'
sys.path.insert(0, str(AGENT_BASE))
SPEC = importlib.util.find_spec('provider')
provider = importlib.import_module('provider') if SPEC else None
import base


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(provider, 'Shared official provider configuration is missing')
        self.protocol = json.loads((AGENT_BASE / 'protocol.json').read_text())

    def test_openai_official_endpoint_and_key(self):
        p = provider.configure_protocol(self.protocol, env={})
        self.assertEqual((p['provider'], p['api'], p['base_url'], p['api_key_env']),
                         ('openai', 'openai-responses', 'https://api.openai.com/v1', 'OPENAI_API_KEY'))
        self.assertNotIn('api_key', p)

    def test_anthropic_model_and_limits(self):
        p = provider.configure_protocol(self.protocol, 'pi', {'MODEL_PROVIDER': 'anthropic'})
        self.assertEqual((p['model'], p['context_window'], p['max_tokens']),
                         ('claude-sonnet-4-6', 200000, 64000))
        self.assertEqual(p['api'], 'anthropic-messages')
        self.assertEqual(p['base_url'], 'https://api.anthropic.com')
        self.assertEqual(p['api_key_env'], 'ANTHROPIC_API_KEY')

    def test_model_and_budget_overrides_do_not_mutate_input(self):
        before = dict(self.protocol)
        p = provider.configure_protocol(self.protocol, env={
            'MODEL_PROVIDER': 'anthropic', 'MODEL_NAME': 'claude-opus-4-6',
            'MODEL_CONTEXT_WINDOW': '1000000', 'MODEL_MAX_OUTPUT_TOKENS': '8192',
            'MODEL_REASONING_EFFORT': 'medium'})
        self.assertEqual((p['model'], p['context_window'], p['max_tokens'], p['reasoning_effort']),
                         ('claude-opus-4-6', 1000000, 8192, 'medium'))
        self.assertEqual(self.protocol, before)

    def test_codex_rejects_anthropic(self):
        with self.assertRaisesRegex(ValueError, 'Codex.*OpenAI'):
            provider.configure_protocol(self.protocol, 'codex', {'MODEL_PROVIDER': 'anthropic'})

    def test_rejects_unknown_provider_and_invalid_limits(self):
        for env in [{'MODEL_PROVIDER': 'other'}, {'MODEL_CONTEXT_WINDOW': '0'},
                    {'MODEL_MAX_OUTPUT_TOKENS': '-1'}, {'MODEL_CONTEXT_WINDOW': 'bad'},
                    {'MODEL_NAME': '  '}, {'MODEL_REASONING_EFFORT': 'unsupported'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                provider.configure_protocol(self.protocol, env=env)

    def test_endpoint_overrides_cannot_redirect_requests(self):
        p = provider.configure_protocol(dict(self.protocol, base_url='https://example.invalid/v1'),
                                        env={'MODEL_BASE_URL': 'https://example.invalid/api',
                                             'OPENAI_BASE_URL': 'https://example.invalid/v1'})
        self.assertEqual(p['base_url'], 'https://api.openai.com/v1')

    def test_missing_key_stops_before_launch(self):
        p = provider.configure_protocol(self.protocol, env={})
        with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
            provider.require_api_key(p, {})
        provider.require_api_key(p, {'OPENAI_API_KEY': 'synthetic-test-value'})

    def test_model_config_contains_environment_name_only(self):
        p = provider.configure_protocol(self.protocol, env={'MODEL_PROVIDER': 'anthropic'})
        cfg = provider.pi_provider_config(p)
        self.assertEqual(cfg['providers']['anthropic']['apiKey'], 'ANTHROPIC_API_KEY')
        self.assertEqual(cfg['providers']['anthropic']['models'][0]['api'], 'anthropic-messages')
        self.assertEqual(provider.configure_protocol(p, env={}), p)

    def test_environment_excludes_alternate_auth(self):
        p = provider.configure_protocol(self.protocol, env={'MODEL_PROVIDER': 'anthropic'})
        env = provider.process_env(p, {'ANTHROPIC_API_KEY': 'synthetic-test-value',
              'OPENAI_API_KEY': 'unrelated-test-value', 'ANTHROPIC_OAUTH_TOKEN': 'oauth-test-value',
              'MODEL_API_KEY': 'legacy-test-value', 'ANTHROPIC_BASE_URL': 'https://example.invalid',
              'PATH': '/usr/bin'})
        self.assertEqual(env['ANTHROPIC_API_KEY'], 'synthetic-test-value')
        self.assertEqual(env['PATH'], '/usr/bin')
        for name in ['OPENAI_API_KEY', 'ANTHROPIC_OAUTH_TOKEN', 'MODEL_API_KEY', 'ANTHROPIC_BASE_URL']:
            self.assertNotIn(name, env)

    def test_error_redaction_for_both_providers(self):
        text = provider.redact('one synthetic-openai-value two synthetic-anthropic-value Bearer bearer-test-value',
              {'OPENAI_API_KEY': 'synthetic-openai-value', 'ANTHROPIC_API_KEY': 'synthetic-anthropic-value'})
        self.assertEqual(text, 'one [redacted] two [redacted] Bearer [redacted]')

    def test_error_redaction_covers_other_inherited_credentials(self):
        text = provider.redact('before synthetic-other-secret after', {'OTHER_SERVICE_TOKEN': 'synthetic-other-secret'})
        self.assertEqual(text, 'before [redacted] after')

    def test_output_budget_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            provider.configure_protocol(dict(self.protocol, max_tokens=1.5), env={})

    def test_base_imports_by_file_as_existing_task_loader_does(self):
        code = ('import importlib.util; from pathlib import Path; '
                's=importlib.util.spec_from_file_location("agent_base", Path("experiments/agent_base/base.py")); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert m.configure_protocol({},env={})["provider"] == "openai"')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pi_command_uses_selected_native_api_and_keeps_sandbox(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(base.os.environ, {
            'MODEL_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': 'synthetic-test-value'}, clear=True):
            attempt = Path(d)
            cmd, env = base.build_command(attempt, {'corpus_dir': d},
                {'harness': 'pi', 'context_level': 'level3'}, self.protocol)
            self.assertEqual(cmd[cmd.index('--provider') + 1], 'anthropic')
            self.assertEqual(cmd[cmd.index('--tools') + 1], 'read,bash')
            self.assertEqual(cmd[0:3], ['/usr/bin/python3', '-S', str(base.LANDLOCK)])
            model_text = (attempt / 'agent/models.json').read_text()
            self.assertNotIn('synthetic-test-value', model_text + json.dumps(cmd))
            self.assertEqual(json.loads(model_text)['providers']['anthropic']['baseUrl'], 'https://api.anthropic.com')
            self.assertEqual(env['ANTHROPIC_API_KEY'], 'synthetic-test-value')

    def test_codex_command_uses_official_responses(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(base.os.environ, {
            'OPENAI_API_KEY': 'synthetic-test-value'}, clear=True):
            cmd, env = base.build_command(Path(d), {'corpus_dir': d}, {'harness': 'codex'}, self.protocol)
            joined = ' '.join(cmd)
            self.assertIn('https://api.openai.com/v1', joined)
            self.assertIn('wire_api="responses"', joined)
            self.assertIn('OPENAI_API_KEY', joined)
            self.assertNotIn('synthetic-test-value', joined)
            self.assertIn('--ignore-user-config', cmd)
            self.assertEqual(env['OPENAI_API_KEY'], 'synthetic-test-value')


if __name__ == '__main__':
    unittest.main()
