"""Closed-book probe prompts (ID probe A0-ID, name probe A0-Name); fixed before the run."""

KB = {
    "amazon": ("the STaRK-Amazon product knowledge base", "products"),
    "mag": ("the STaRK-MAG academic knowledge base built from the Microsoft Academic Graph", "papers"),
    "prime": ("the STaRK-Prime biomedical knowledge base built from PrimeKG", "entities of any of these types: disease, gene/protein, molecular_function, drug, pathway, anatomy, effect/phenotype, biological_process, cellular_component, exposure"),
}


def prompt_for(row, probe):
    kb, answers = KB[row["dataset"]]
    text = ("Answer the retrieval query from your own knowledge. You do not have access to the knowledge base, its files, or any tools.\n"
            f"Knowledge base: {kb}. Valid answers: {answers}.\n"
            "The query may have several correct answers.\n")
    if probe == "A0-ID":
        text += ("Return the STaRK benchmark node_id integers of the correct answers. If you remember them, list them; otherwise give your best guesses.\n"
                 "Return 20 unique integer IDs ranked by confidence, most confident first. Return fewer than 20 only if you cannot give that many.\n"
                 'Return only JSON with exactly this structure: {"node_ids": [123, 456]}. Do not include Markdown, explanations, or citations outside the JSON.\n\n')
    elif probe == "A0-Name" and row["dataset"] in {"mag", "prime"}:
        kind = "exact paper titles" if row["dataset"] == "mag" else "exact entity names as used in PrimeKG"
        text += (f"Return the {kind} of the correct answers.\n"
                 "Return 20 unique names ranked by confidence, most confident first. Return fewer than 20 only if you cannot give that many.\n"
                 'Return only JSON with exactly this structure: {"names": ["...", "..."]}. Do not include Markdown, explanations, or citations outside the JSON.\n\n')
    else:
        raise ValueError("Unknown probe/dataset combination")
    return text + f"Query: {row['query']}\n"
