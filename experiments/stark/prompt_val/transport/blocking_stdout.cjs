"use strict";

const mode = process.argv.indexOf("--mode");
if (process.argv[1]?.endsWith("/coding-agent/dist/cli.js") &&
    mode !== -1 && process.argv[mode + 1] === "rpc") {
  // The RPC emitter ignores write(false); block before it captures stdout.
  // This preserves every event while the Python recorder catches up.
  if (typeof process.stdout._handle?.setBlocking !== "function") {
    throw new Error("Pi RPC stdout does not support blocking writes");
  }
  process.stdout._handle.setBlocking(true);
}
