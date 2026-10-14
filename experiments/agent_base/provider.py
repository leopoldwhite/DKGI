"""Official API configuration; Pi's maintained SDK adapters own the wire protocols."""

import os
import re


PROVIDERS = {
    "openai": ("openai-responses", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.4", 272000, 128000),
    "anthropic": ("anthropic-messages", "https://api.anthropic.com", "ANTHROPIC_API_KEY", "claude-sonnet-4-6", 200000, 64000),
}
REASONING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}


def configure_protocol(protocol, harness=None, env=None):
    """Return non-secret settings, with explicit environment overrides and official endpoints."""
    env = os.environ if env is None else env
    previous = protocol.get("provider", "openai")
    name = env.get("MODEL_PROVIDER", previous).strip().lower()
    if name not in PROVIDERS:
        raise ValueError("MODEL_PROVIDER must be openai or anthropic")
    if harness not in (None, "pi", "codex"):
        raise ValueError("Harness must be pi or codex")
    if harness == "codex" and name != "openai":
        raise ValueError("Codex supports only OpenAI Responses; use Pi for Anthropic")
    api, url, key_env, default_model, context, output = PROVIDERS[name]
    same_provider = name == previous
    result = dict(protocol)
    model = env.get("MODEL_NAME", protocol.get("model", default_model) if same_provider else default_model).strip()
    if not model or any(c.isspace() for c in model):
        raise ValueError("MODEL_NAME must be a nonempty API model ID without whitespace")
    result.update(provider=name, api=api, base_url=url, api_key_env=key_env, model=model)
    for key, variable, default in (("context_window", "MODEL_CONTEXT_WINDOW", context),
                                    ("max_tokens", "MODEL_MAX_OUTPUT_TOKENS", output)):
        value = env.get(variable, protocol.get(key, default) if same_provider else default)
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{variable} must be a positive integer") from None
        if isinstance(value, (bool, float)) or number <= 0:
            raise ValueError(f"{variable} must be a positive integer")
        result[key] = number
    effort = env.get("MODEL_REASONING_EFFORT", protocol.get("reasoning_effort", "high")).strip().lower()
    if effort not in REASONING_LEVELS:
        raise ValueError("MODEL_REASONING_EFFORT must be off, minimal, low, medium, high or xhigh")
    result["reasoning_effort"] = effort
    # Credentials are environment-only even if a caller supplied an obsolete field.
    for key in ("api_key", "apiKey", "token", "password", "secret"):
        result.pop(key, None)
    return result


def model_definition(protocol):
    """Pi model metadata; cost is omitted rather than inventing provider prices."""
    p = configure_protocol(protocol, env={})
    return {"id": p["model"], "name": p["model"], "api": p["api"],
            "reasoning": p["reasoning_effort"] != "off", "input": ["text"],
            "contextWindow": p["context_window"], "maxTokens": p["max_tokens"]}


def pi_provider_config(protocol):
    p = configure_protocol(protocol, "pi", env={})
    return {"providers": {p["provider"]: {"baseUrl": p["base_url"], "apiKey": p["api_key_env"],
            "api": p["api"], "models": [model_definition(p)]}}}


def require_api_key(protocol, env=None):
    """Fail before launch; never return, print or persist a credential."""
    env = os.environ if env is None else env
    name = configure_protocol(protocol, env={})["api_key_env"]
    if not env.get(name, "").strip():
        raise ValueError(f"{name} is not set")


def process_env(protocol, env=None):
    """Pass only the selected provider credential and prevent alternate auth/endpoint fallbacks."""
    env = dict(os.environ if env is None else env)
    selected = configure_protocol(protocol, env={})["api_key_env"]
    for name in list(env):
        if name != selected and re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I):
            env.pop(name)
    for name in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "MODEL_BASE_URL"):
        env.pop(name, None)
    return env


def redact(text, env=None):
    env = os.environ if env is None else env
    text = str(text)
    for name, value in env.items():
        if value and re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I):
            text = text.replace(value, "[redacted]")
    return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[redacted]", text)


def total_cost(rows, key="attempt_cost_usd"):
    """A full total is unknown when any component has an unknown price."""
    values = [row.get(key) for row in rows]
    return None if any(value is None for value in values) else sum(values)
