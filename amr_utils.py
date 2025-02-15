"""
This file has utils functions to extract an AMR graph from a sentence.
It is based on the code from https://github.com/zzshou/AMRSim/blob/main/preprocess/utils.py
"""

import logging

import amrlib
import penman

# Suppress INFO, DEBUG and WARNING logs from penman
logging.getLogger("penman.layout").setLevel(logging.ERROR)


def simplify_amr_tokens(tokens, v2c, out_graph_type="bipartite"):
    """
    Simplifies the linearized AMR by processing the tokens and mapping variables to concepts.

    Args:
        tokens (list): Linearized AMR tokens.
        v2c (dict): A mapping of variables to their corresponding concepts.

    Returns:
        tuple: A list of simplified tokens and a mapping of variables to their token positions.
    """
    mapping = {}  # Mapping from variable names to positions in the token list
    new_tokens = []  # Simplified token list
    save_map = (
        False  # Indicates when to save the mapping from variable to token position
    )
    last_map = None  # Stores the last variable to map
    for tok in tokens:
        # ignore instance-of
        if tok.startswith("("):  # Ignore variable names in tokens
            last_map = tok.replace("(", "")
            continue
        elif tok == "/":  # Indicates that the following token is a concept
            save_map = True
            continue
        elif out_graph_type == "multirelational" and tok.startswith(":"):
            continue  # Ignore relations in multirelational graphs
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(":"):  # Process relations (e.g., :ARG0)
            new_tok = tok.strip(")").split("~")[0]
            new_tokens.append(new_tok)
        else:  # Process concepts and reentrancies
            new_tok = tok.strip(")").split("~")[0]

            if new_tok == "":
                continue

            # now we check if it is a concept or a variable (reentrancy)
            if new_tok in v2c:  # If it's a reentrant variable
                if new_tok not in mapping:
                    mapping[new_tok] = set()
                mapping[new_tok].add(len(new_tokens))
                if v2c[new_tok] is not None:
                    new_tok = v2c[new_tok]

            elif new_tok.isnumeric():  # If it's a number, keep it as is
                new_tok = new_tok

            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':  # Remove quotes from strings
                new_tok = new_tok[1:-1]

            if new_tok != "":
                new_tokens.append(new_tok)

            if save_map:  # Save the variable-to-position mapping
                if last_map not in mapping:
                    mapping[last_map] = set()
                mapping[last_map].add(len(new_tokens) - 1)
                save_map = False

    return new_tokens, mapping


def find_token_positions(new_tokens, src):
    """
    Retrieves the positions of a specific token in the list of tokens.

    Args:
        new_tokens (list): Simplified tokens.
        src (str): The token to search for.

    Returns:
        list: A list of positions where the token appears.
    """
    pos = []
    for idx, n in enumerate(new_tokens):
        if n == src:
            pos.append(idx)
    return pos


def generate_bipartite_graph_graph(graph, new_tokens, mapping, roles_in_order):
    """
    Converts the AMR graph into a set of bipartite edge triples and node tokens.

    Args:
        graph (penman.Graph): The decoded AMR graph.
        new_tokens (list): Simplified tokens.
        mapping (dict): Mapping of variables to token positions.
        roles_in_order (list): List of roles (relations) in the order they appear.

    Returns:
        tuple: Node tokens and bipartite edge triples.
    """
    triples = []  # Stores the final edge triples
    nodes_to_print = new_tokens  # Final node tokens

    graph_triples = graph.triples  # Extracted triples from the graph

    edge_id = -1  # Initialize edge identifier
    triples_set = set()  # To avoid duplicate triples
    count_roles = 0  # Index to iterate through roles
    for triple in graph_triples:
        src, edge, tgt = triple
        if edge == ":instance" or edge == ":instance-of":
            continue

        # if penman.layout.appears_inverted(graph_penman, v):
        # Handle "-of" inversions for certain roles
        if (
            "-of" in roles_in_order[count_roles]
            and "-off" not in roles_in_order[count_roles]
        ):
            if edge != ":consist-of":
                edge = edge + "-of"
                old_tgt = tgt
                tgt = src
                src = old_tgt

        assert roles_in_order[count_roles] == edge
        count_roles += 1

        if edge == ":wiki":
            continue

        src = str(src).replace('"', "")
        tgt = str(tgt).replace('"', "")

        if src not in mapping:
            src_id = find_token_positions(new_tokens, src)
        else:
            src_id = sorted(list(mapping[src]))

        edge_id = find_edge_position(new_tokens, edge, edge_id)

        if tgt not in mapping:
            tgt_id = find_token_positions(new_tokens, tgt)
        else:
            tgt_id = sorted(list(mapping[tgt]))

        for s_id in src_id:
            if (s_id, edge_id, "d") not in triples_set:
                triples.append((s_id, edge_id, "d"))
                triples_set.add((s_id, edge_id, "d"))
                triples.append((edge_id, s_id, "r"))
        for t_id in tgt_id:
            if (edge_id, t_id, "d") not in triples_set:
                triples.append((edge_id, t_id, "d"))
                triples_set.add((edge_id, t_id, "d"))
                triples.append((t_id, edge_id, "r"))

    if nodes_to_print == []:
        # single node graph, first triple is ":top", second triple is the node
        triples.append((0, 0, "s"))
    return nodes_to_print, triples


def find_edge_position(tokens, edge, edge_id):
    """
    Finds the position of an edge (relation) in the token list.

    Args:
        tokens (list): Simplified tokens.
        edge (str): The edge (relation) to find.
        edge_id (int): Current edge identifier.

    Returns:
        int: The position of the edge in the token list.
    """
    for idx in range(edge_id + 1, len(tokens)):
        if tokens[idx] == edge:
            return idx


def map_variables_to_concepts(graph_penman):
    """
    Creates a dictionary mapping from variables to their corresponding concepts in the AMR graph.

    Args:
        graph_penman (penman.Graph): The decoded AMR graph.

    Returns:
        dict: A mapping of variables to their concepts.
    """
    instances = graph_penman.instances()
    dict_insts = {}
    for i in instances:
        dict_insts[i.source] = i.target
    return dict_insts


def convert_amr_to_graph(amr_str, out_graph_type="bipartite", rel_glossary=None):
    """
    Simplifies an AMR string into node tokens and bipartite edge triples.

    Args:
        amr_str (str): The AMR string.

    Returns:
        tuple: Simplified node tokens and bipartite edge triples.
    """
    # Decode AMR string into a Penman graph
    graph_penman = penman.decode(amr_str)
    v2c_penman = map_variables_to_concepts(graph_penman)

    # Linearize and clean up the AMR
    linearized_amr = penman.encode(graph_penman).replace("\t", "").replace("\n", "")
    tokens = linearized_amr.split()

    # Simplify the tokens
    new_tokens, mapping = simplify_amr_tokens(tokens, v2c_penman, out_graph_type)

    # Extract roles in the order they appear
    roles_in_order = [
        token for token in tokens if token.startswith(":") and token != ":instance-of"
    ]

    # Generate node tokens and edge triples
    if out_graph_type == "bipartite":
        nodes, triples = generate_bipartite_graph_graph(
            graph_penman, new_tokens, mapping, roles_in_order
        )
        triples = sorted(triples)
        return nodes, triples
    elif out_graph_type == "multirelational":
        nodes, triples, edge_types = generate_multirelational_graph(
            graph_penman, new_tokens, mapping, roles_in_order, rel_glossary
        )
        return nodes, triples, edge_types
    else:
        raise ValueError("Invalid graph type: {}".format(out_graph_type))


def generate_multirelational_graph(
    graph, new_tokens, mapping, roles_in_order, rel_glossary
):
    """
    Converts the AMR graph into a set of multirelational edge triples and node tokens.

    Args:
        graph (penman.Graph): The decoded AMR graph.
        new_tokens (list): Simplified tokens.
        mapping (dict): Mapping of variables to token positions.
        roles_in_order (list): List of roles (relations) in the order they appear.

    Returns:
        tuple: Node tokens and multirelational edge triples.
    """
    assert (
        rel_glossary is not None
    ), "Relation glossary is required for multirelational graphs"

    triples = []  # Stores the final multirelational edge triples
    new_tokens = [
        token for token in new_tokens if not token.startswith(":")
    ]  # node tokens without relations

    graph_triples = graph.triples  # Extracted triples from the graph
    count_roles = 0  # Index to iterate through roles
    edge_types = []  # Stores the edge types

    for triple in graph_triples:
        src, edge, tgt = triple
        if edge == ":instance" or edge == ":instance-of":
            continue

        # Handle "-of" inversions for certain roles
        if (
            "-of" in roles_in_order[count_roles]
            and "-off" not in roles_in_order[count_roles]
        ):
            if edge != ":consist-of":
                edge = edge + "-of"
                src, tgt = tgt, src

        assert roles_in_order[count_roles] == edge
        count_roles += 1

        if edge in [":wiki", ":"]:
            continue

        src = str(src).replace('"', "")
        tgt = str(tgt).replace('"', "")

        # Get the position(s) of the source and target nodes
        src_ids = (
            find_token_positions(new_tokens, src)
            if src not in mapping
            else sorted(list(mapping[src]))
        )
        tgt_ids = (
            find_token_positions(new_tokens, tgt)
            if tgt not in mapping
            else sorted(list(mapping[tgt]))
        )

        try:
            # Get the id of the edge (relation)
            edge_type = rel_glossary.index(edge)
        except:
            breakpoint()

        # Generate multirelational triples (src, edge, tgt)
        for s_id in src_ids:
            for t_id in tgt_ids:
                triples.append((s_id, t_id, 0))
                edge_types.append(edge_type)

    if not new_tokens:
        # Handle single-node graphs (if applicable)
        triples.append((0, 0, 0))
        edge_types.append(0)

    assert len(edge_types) == len(triples)

    return new_tokens, triples, edge_types


def extract_amr_graphs_from_sentences(sentences):
    # Load pretrained StoG model and generate AMR
    stog = amrlib.load_stog_model(
        model_dir="model_parse_xfm_bart_large-v0_1_0", device="cuda:0", batch_size=4
    )
    if isinstance(sentences, str):
        sentences = [sentences]
    gen_amr_strs = stog.parse_sents(sentences)
    nodes_batch, triples_batch = [], []
    for amr_str in gen_amr_strs:
        gen_amr_str = amr_str.split("\n", 1)[1]

        # Convert String AMR to list of nodes and triples
        amr = convert_amr_to_graph(gen_amr_str)
        if amr is None:
            return None

        nodes, triples = amr

        nodes_batch.append(nodes)
        triples_batch.append(triples)

    return nodes_batch, triples_batch


if __name__ == "__main__":
    rel_glossary = [
        ":ARG0",
        ":ARG0-of",
        ":ARG1",
        ":time",
        ":op1",
        ":name",
        ":op2",
        ":ARG2-of",
        ":ARG3",
        ":quant",
        ":domain",
        ":ARG1-of",
    ]
    # amr1 = "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"
    # expected_bipartite_node_tokens1 = [
    #     "prospect-02",
    #     ":ARG0",
    #     "person",
    #     ":ARG0-of",
    #     "engineer-01",
    #     ":ARG1",
    #     "mechanics",
    #     ":ARG1",
    #     "amr-unknown",
    #     ":time",
    #     "future",
    # ]
    # expected_bipartite_edge_triples1 = [
    #     [0, 1, 0],
    #     [0, 7, 0],
    #     [0, 9, 0],
    #     [1, 0, 1],
    #     [1, 2, 0],
    #     [2, 1, 1],
    #     [2, 3, 0],
    #     [3, 2, 1],
    #     [3, 4, 0],
    #     [4, 3, 1],
    #     [4, 5, 0],
    #     [5, 4, 1],
    #     [5, 6, 0],
    #     [6, 5, 1],
    #     [7, 0, 1],
    #     [7, 8, 0],
    #     [8, 7, 1],
    #     [9, 0, 1],
    #     [9, 10, 0],
    #     [10, 9, 1],
    # ]
    # actual_bipartite_node_tokens1, actual_bipartite_edge_triples1 = (
    #     convert_amr_to_graph(amr1, out_graph_type="bipartite")
    # )
    # actual_bipartite_edge_triples1 = [
    #     [from_idx, to_idx, 0 if relation == "d" else 1]
    #     for from_idx, to_idx, relation in actual_bipartite_edge_triples1
    # ]
    # assert (
    #     actual_bipartite_node_tokens1 == expected_bipartite_node_tokens1
    # ), "Bipartite node tokens #1:\nactual: {}, expected: {}".format(
    #     actual_bipartite_node_tokens1, expected_bipartite_node_tokens1
    # )
    # assert (
    #     actual_bipartite_edge_triples1 == expected_bipartite_edge_triples1
    # ), "Bipartite node tokens #1:\nactual: {}, expected: {}".format(
    #     actual_bipartite_edge_triples1, expected_bipartite_edge_triples1
    # )

    # expected_multirelational_node_tokens1 = [
    #     "prospect-02",
    #     "person",
    #     "engineer-01",
    #     "mechanics",
    #     "amr-unknown",
    #     "future",
    # ]
    # expected_multirelational_edge_triples1 = [
    #     (0, 1, 0),
    #     (0, 4, 0),
    #     (0, 5, 0),
    #     (1, 2, 0),
    #     (2, 3, 0),
    # ]
    # expected_multirelational_edge_types1 = [0, 1, 2, 2, 3]

    # (
    #     actual_multirelational_node_tokens1,
    #     actual_multirelational_edge_triples1,
    #     actual_multirelational_edge_types1,
    # ) = convert_amr_to_graph(
    #     amr1, out_graph_type="multirelational", rel_glossary=rel_glossary
    # )
    # assert (
    #     actual_multirelational_node_tokens1 == expected_multirelational_node_tokens1
    # ), "Multirelational node tokens #1:\nactual: {}, expected: {}".format(
    #     actual_multirelational_node_tokens1, expected_multirelational_node_tokens1
    # )
    # assert (
    #     actual_multirelational_edge_triples1 == expected_multirelational_edge_triples1
    # ), "Multirelational node tokens #1:\nactual: {}, expected: {}".format(
    #     actual_multirelational_edge_triples1, expected_multirelational_edge_triples1
    # )
    # assert (
    #     actual_multirelational_edge_types1 == expected_multirelational_edge_types1
    # ), "Multirelational node tokens #1:\nactual: {}, expected: {}".format(
    #     actual_multirelational_edge_types1, expected_multirelational_edge_types1
    # )

    # amr2 = '(u / useful-05\n      :ARG1 (l / learn-01\n            :ARG1 (a / amr-choice\n                  :op1 (l2 / language\n                        :name (n / name\n                              :op1 "German"))\n                  :op2 (l3 / language\n                        :name (n2 / name\n                              :op1 "French"))))\n      :ARG2-of (h / have-degree-91\n            :ARG1 l\n            :ARG3 (m / more)))'
    # expected_bipartite_node_tokens2 = [
    #     "useful-05",
    #     ":ARG1",
    #     "learn-01",
    #     ":ARG1",
    #     "amr-choice",
    #     ":op1",
    #     "language",
    #     ":name",
    #     "name",
    #     ":op1",
    #     "German",
    #     ":op2",
    #     "language",
    #     ":name",
    #     "name",
    #     ":op1",
    #     "French",
    #     ":ARG2-of",
    #     "have-degree-91",
    #     ":ARG1",
    #     "learn-01",
    #     ":ARG3",
    #     "more",
    # ]
    # expected_bipartite_edge_triples2 = [
    #     [0, 1, 0],
    #     [0, 17, 0],
    #     [1, 0, 1],
    #     [1, 2, 0],
    #     [1, 20, 0],
    #     [2, 1, 1],
    #     [2, 3, 0],
    #     [2, 19, 1],
    #     [3, 2, 1],
    #     [3, 4, 0],
    #     [3, 20, 1],
    #     [4, 3, 1],
    #     [4, 5, 0],
    #     [4, 11, 0],
    #     [5, 4, 1],
    #     [5, 6, 0],
    #     [6, 5, 1],
    #     [6, 7, 0],
    #     [7, 6, 1],
    #     [7, 8, 0],
    #     [8, 7, 1],
    #     [8, 9, 0],
    #     [9, 8, 1],
    #     [9, 10, 0],
    #     [10, 9, 1],
    #     [11, 4, 1],
    #     [11, 12, 0],
    #     [12, 11, 1],
    #     [12, 13, 0],
    #     [13, 12, 1],
    #     [13, 14, 0],
    #     [14, 13, 1],
    #     [14, 15, 0],
    #     [15, 14, 1],
    #     [15, 16, 0],
    #     [16, 15, 1],
    #     [17, 0, 1],
    #     [17, 18, 0],
    #     [18, 17, 1],
    #     [18, 19, 0],
    #     [18, 21, 0],
    #     [19, 2, 0],
    #     [19, 18, 1],
    #     [19, 20, 0],
    #     [20, 1, 1],
    #     [20, 3, 0],
    #     [20, 19, 1],
    #     [21, 18, 1],
    #     [21, 22, 0],
    #     [22, 21, 1],
    # ]
    # actual_bipartite_node_tokens2, actual_bipartite_edge_triples2 = (
    #     convert_amr_to_graph(amr2, out_graph_type="bipartite")
    # )
    # expected_multirelational_node_tokens2 = [
    #     "useful-05",
    #     "learn-01",
    #     "amr-choice",
    #     "language",
    #     "name",
    #     "German",
    #     "language",
    #     "name",
    #     "French",
    #     "have-degree-91",
    #     "learn-01",
    #     "more",
    # ]
    # expected_multirelational_edge_triples2 = [
    #     [0, 1, 0],  # useful-05 -> learn-01 via :ARG1
    #     [0, 9, 0],  # useful-05 -> have-degree-91 via :ARG2-of
    #     [1, 2, 0],  # learn-01 -> amr-choice via :ARG1
    #     [1, 11, 0],  # learn-01 -> more via :ARG3 (nested via have-degree-91)
    #     [2, 3, 0],  # amr-choice -> language via :op1
    #     [2, 6, 0],  # amr-choice -> language via :op2
    #     [3, 4, 0],  # language -> name via :name
    #     [4, 5, 0],  # name -> German via :op1
    #     [6, 7, 0],  # language -> name via :name (second language node)
    #     [7, 8, 0],  # name -> French via :op1 (for French)
    #     [9, 1, 0],  # have-degree-91 -> learn-01 via :ARG1 (reverse link)
    # ]
    # actual_multirelational_node_tokens2, actual_multirelational_edge_triples2 = (
    #     convert_amr_to_graph(amr2, out_graph_type="multirelational")
    # )
    # assert actual_bipartite_edge_triples2 == expected_bipartite_node_tokens2
    # assert actual_bipartite_node_tokens2 == expected_bipartite_edge_triples2
    # assert actual_multirelational_edge_triples2 == expected_multirelational_node_tokens2
    # assert actual_multirelational_node_tokens2 == expected_multirelational_edge_triples2

    amr3 = '(d / do-02\n      :ARG0 (p / person\n            :name (n / name\n                  :op1 "Rahul"\n                  :op2 "Gandhi"))\n      :ARG1 (t / thing\n            :quant (s / some)\n            :ARG1-of (g / good-02)\n            :domain (a / amr-unknown)))'

    expected_multirelational_node_tokens3 = [
        "do-02",
        "person",
        "name",
        "Rahul",
        "Gandhi",
        "thing",
        "some",
        "good-02",
        "amr-unknown",
    ]
    expected_multirelational_edge_triples3 = [
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
        (2, 4, 0),
        (0, 5, 0),
        (5, 6, 0),
        (5, 7, 0),
        (5, 8, 0),
    ]
    expected_multirelational_edge_types3 = [0, 5, 4, 6, 2, 9, 11, 10]

    (
        actual_multirelational_node_tokens3,
        actual_multirelational_edge_triples3,
        actual_multirelational_edge_types3,
    ) = convert_amr_to_graph(
        amr3, out_graph_type="multirelational", rel_glossary=rel_glossary
    )
    assert (
        actual_multirelational_node_tokens3 == expected_multirelational_node_tokens3
    ), "Multirelational node tokens #3:\nactual: {}, expected: {}".format(
        actual_multirelational_node_tokens3, expected_multirelational_node_tokens3
    )
    assert (
        actual_multirelational_edge_triples3 == expected_multirelational_edge_triples3
    ), "Multirelational node tokens #3:\nactual: {}, expected: {}".format(
        actual_multirelational_edge_triples3, expected_multirelational_edge_triples3
    )
    assert (
        actual_multirelational_edge_types3 == expected_multirelational_edge_types3
    ), "Multirelational node tokens #3:\nactual: {}, expected: {}".format(
        actual_multirelational_edge_types3, expected_multirelational_edge_types3
    )
