"""KB client for the preconnected, sandbox-preserved descriptor."""

import json
import os

from _kbtools_base import Client as OriginalClient


class Client(OriginalClient):
    def call(self, name, **arguments):
        # One connection per shell command; composed calls are serialized.
        data = (json.dumps({"name": name, "arguments": arguments}) + "\n").encode()
        while data:
            data = data[os.write(3, data):]
        with os.fdopen(os.dup(3), "rb") as source:
            response = json.loads(source.readline())
        if not response["ok"]:
            raise ValueError(response["error"])
        return response["text"]
