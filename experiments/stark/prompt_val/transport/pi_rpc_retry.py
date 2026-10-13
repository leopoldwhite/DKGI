"""Keep the RPC runner alive while Pi retries a failed model request."""

from __future__ import annotations

import re
import sys

from dci.benchmark import pi_rpc_runner as original


# Match the AgentSession._isRetryableError, without retrying hard failures.
RETRYABLE = re.compile(
    r"overloaded|provider.?returned.?error|rate.?limit|too many requests|429|500|502|503|504|"
    r"service.?unavailable|server.?error|internal.?error|network.?error|connection.?error|"
    r"connection.?refused|other side closed|fetch failed|upstream.?connect|reset before headers|"
    r"socket hang up|timed? out|timeout|terminated|retry delay", re.I,
)
FATAL = re.compile(
    r"\b(?:401|402|403)\b|invalid[_ ]api[_ ]key|incorrect api key|insufficient[_ ]quota|"
    r"model[_ ]not[_ ]found|invalid model|JSONDecodeError|SyntaxError|invalid.?json|protocol error",
    re.I,
)
OVERFLOW = re.compile(
    r"prompt (?:is )?too long|request_too_large|input is too long|context[_ ]length[_ ]exceeded|"
    r"exceeds the context window|maximum (?:context|prompt) length|context window exceeds limit|"
    r"exceeded model token limit|exceeds the available context size|greater than the context length|"
    r"model_context_window_exceeded|input token count.*exceeds the maximum|"
    r"reduce the length of the messages|exceeds the limit of \d+|too many tokens|"
    r"token limit exceeded|too large for model with \d+ maximum context length|"
    r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.I,
)
NON_OVERFLOW = re.compile(r"^(?:Throttling error|Service unavailable):|rate limit|too many requests", re.I)


class PiRpcRetryClient(original.PiRpcClient):
    def prompt_and_wait(self, message, *, recorder=None, max_turns=None):
        settings = original.read_json_if_exists(self.agent_dir / "settings.json") or {}
        retry_enabled = settings.get("retry", {}).get("enabled", True)
        request_id = self._next_id()
        self._send({"id": request_id, "type": "prompt", "message": message})
        prompt_ack = False
        seen_turns = 0
        sent_turn_limit_abort = False

        while True:
            event = self._read_json_line()
            if recorder:
                recorder.record_event(event)
            event_type = event.get("type")
            if event_type == "response":
                if event.get("id") == request_id:
                    if not event.get("success", False):
                        raise RuntimeError(f"RPC prompt failed: {event.get('error', 'unknown error')}")
                    prompt_ack = True
                continue
            if event_type == "turn_start":
                seen_turns += 1
                if max_turns is not None and seen_turns > max_turns and not sent_turn_limit_abort:
                    abort_id = self._next_id()
                    self._send({"id": abort_id, "type": "abort"})
                    sent_turn_limit_abort = True
                    note = f"Reached max_turns={max_turns}; sent RPC abort before turn {seen_turns}."
                    if recorder:
                        recorder.add_note(note)
                    sys.stderr.write(f"\n[runner] {note}\n")
                    sys.stderr.flush()
                continue
            if event_type == "message_update":
                assistant_event = event.get("assistantMessageEvent", {})
                if assistant_event.get("type") == "text_delta":
                    sys.stdout.write(assistant_event.get("delta", ""))
                    sys.stdout.flush()
                continue
            if event_type in {"tool_execution_start", "tool_execution_end"} and self.show_tools:
                phase = "start" if event_type == "tool_execution_start" else "end"
                suffix = "" if phase == "start" else f" error={'yes' if event.get('isError') else 'no'}"
                sys.stderr.write(f"\n[tool:{phase}] {event.get('toolName', 'unknown')}{suffix}\n")
                sys.stderr.flush()
                continue
            if event_type == "auto_retry_end" and not event.get("success", False):
                raise RuntimeError(event.get("finalError") or "Pi native retries exhausted")
            if event_type != "agent_end":
                continue
            if not prompt_ack:
                raise RuntimeError("Received agent_end before prompt acknowledgement")
            for final_message in reversed(event.get("messages", [])):
                if final_message.get("role") != "assistant":
                    continue
                reason = final_message.get("stopReason")
                if reason in {"error", "aborted"}:
                    error = final_message.get("errorMessage") or f"Assistant response {reason}"
                    overflow = OVERFLOW.search(error) and not NON_OVERFLOW.search(error)
                    if reason == "error" and retry_enabled and not sent_turn_limit_abort and RETRYABLE.search(error) and not FATAL.search(error) and not overflow:
                        # AgentSession emits auto_retry_start after this agent_end.
                        # Let it resume the same context; never submit another prompt.
                        break
                    raise RuntimeError(error)
                content = final_message.get("content", [])
                if reason == "toolUse" or any(block.get("type") == "toolCall" for block in content):
                    raise RuntimeError("Agent ended with a tool call instead of a final answer")
                final_text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text").strip()
                if not final_text:
                    raise RuntimeError("Agent ended without a final assistant answer")
                return final_text
            else:
                raise RuntimeError("Agent ended without a final assistant answer")


if __name__ == "__main__":
    original.PiRpcClient = PiRpcRetryClient
    raise SystemExit(original.main())
