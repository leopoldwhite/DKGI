import net from "node:net";

function request(name, arguments_) {
  return new Promise((resolve, reject) => {
    const connection = net.createConnection(process.env.DCI_KB_SOCKET);
    let data = "";
    connection.setTimeout(120000, () => connection.destroy(new Error("Tool socket timeout")));
    connection.on("error", reject);
    connection.on("connect", () => connection.write(JSON.stringify({name, arguments: arguments_}) + "\n"));
    connection.on("data", chunk => {
      data += chunk.toString();
      if (!data.includes("\n")) return;
      connection.end();
      try {
        const value = JSON.parse(data);
        resolve({content: [{type: "text", text: value.ok ? value.text : value.error}], isError: !value.ok});
      } catch (error) { reject(error); }
    });
  });
}

export default function(pi) {
  const integer = {type: "integer", minimum: 0};
  const text = {type: "string"};
  const definitions = [
    ["search", "Global ARK BM25 search; returns node IDs, names, types, and summary context.",
      {query: text, k: {type: "integer", minimum: 1, maximum: 30, default: 5}}, ["query"]],
    ["get_node", "Read a complete node record, including attributes and is_candidate.", {node_id: integer}, ["node_id"]],
    ["get_schema", "Read knowledge-base entity types, relation types, and graph interpretation.", {}, []],
  ];
  if (process.env.DCI_KB_ARM !== "A1") definitions.push(
    ["neighbors", "ARK one-hop neighborhood search with optional relation, node type, and keyword filters.",
      {node_id: integer, relation: text, node_type: text, query: text,
       k: {type: "integer", minimum: 1, maximum: 20, default: 20}}, ["node_id"]]);
  for (const [name, description, properties, required] of definitions) pi.registerTool({
    name, label: name, description,
    parameters: {type: "object", properties, required, additionalProperties: false},
    async execute(_id, args) { return request(name, args); },
  });
}
