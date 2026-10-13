"""Standard-library client for the supplied knowledge-base socket."""

import json
import socket


class Client:
    def __init__(self, path):
        self.path = str(path)

    def call(self, name, **arguments):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(120)
            connection.connect(self.path)
            connection.sendall((json.dumps({"name": name, "arguments": arguments}) + "\n").encode())
            with connection.makefile("rb") as source:
                response = json.loads(source.readline())
        if not response["ok"]:
            raise ValueError(response["error"])
        return response["text"]

    def search(self, query, k=5):
        return self.call("search", query=query, k=k)

    def neighbors(self, node_id, relation="", node_type="", query="", k=20):
        return self.call("neighbors", node_id=node_id, relation=relation, node_type=node_type, query=query, k=k)

    def get_node(self, node_id):
        return self.call("get_node", node_id=node_id)

    def get_schema(self):
        return self.call("get_schema")
