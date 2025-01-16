from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from amr_utils import convert_amr_to_graph
from glossary import DEPENDENCIES
from linearize import linearize


def normalize_code(cond, cls):
    cond = "_".join(["CODE", str(cls), str(cond)])
    return cond.replace(" ", "_").upper()


def check_all_positive_ints(lst):
    if isinstance(lst, int):
        return lst >= 0
    elif isinstance(lst, list):
        return all(check_all_positive_ints(element) for element in lst)
    else:
        return False


def standard_processing_function(
    tokenizer,
    max_source_length,
    max_target_length,
    use_linearized_graph=False,
    code_columns=None,
    inference_codes=None,
):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        if use_linearized_graph:
            # requires examples to be single entry
            assert not isinstance(
                inputs, list
            ), f"You have a batch of {len(inputs)} examples. \
                The linearization of the graph is done instance by instance. \
                Please do not use `batched` examples."

            inputs_list = linearize(chain_dep_trees(examples["src_dep_trees"])).split()

            model_inputs = tokenizer(
                inputs_list,
                max_length=max_source_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
                is_split_into_words=True,
            )
        else:
            if code_columns is not None:
                if inference_codes is None:
                    codes = [examples[column] for column in code_columns]
                else:
                    assert len(code_columns) == len(inference_codes)
                    codes = [
                        [int(inference_code)] * len(examples[column])
                        for inference_code, column in zip(inference_codes, code_columns)
                    ]
                codes = list(zip(*codes))
                codes = [
                    " ".join(
                        normalize_code(cond, col) for cond, col in zip(c, code_columns)
                    )
                    for c in codes
                ]
                inputs = [c + inp for inp, c in zip(inputs, codes)]

            model_inputs = tokenizer(
                inputs,
                max_length=max_source_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function


def processing_function_for_text_graph(
    tokenizer,
    max_source_length,
    max_target_length,
    graph_representation="multirelational",
    use_linearized_graph=False,
):
    def preprocess_function(examples):
        # transliterate Unicode text into its closest ASCII representation
        inputs = examples["source"]
        targets = examples["target"]

        # requires examples to be single entry
        assert not isinstance(
            inputs, list
        ), f"You have a batch of {len(inputs)} examples. \
            The linearization of the graph is done instance by instance. \
            Please do not use `batched` examples."

        if use_linearized_graph:
            inputs_list = linearize(chain_dep_trees(examples["src_dep_trees"])).split()
        else:
            inputs_list = [
                node["token"] for node in chain.from_iterable(examples["src_dep_trees"])
            ]

        model_inputs = tokenizer(
            inputs_list,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        if graph_representation == "multirelational":
            graph_inputs = create_multirelational_graph_inputs(examples, model_inputs)
        elif graph_representation == "bipartite":
            graph_inputs = create_bipartite_graph_inputs(
                examples,
                model_inputs,
                tokenizer,
                input_is_linearized=use_linearized_graph,
            )
        else:
            raise ValueError(f"Unknown graph type: {graph_representation}")

        output_dict = {**model_inputs, **graph_inputs}

        # multirelational graphs should also contain edge_type
        assert all(
            key in output_dict
            for key in [
                "input_ids",
                "labels",
                "x",
                "edge_index",
            ]
        ), "Not all items in the list are keys of the dictionary."

        return output_dict

    return preprocess_function


def create_multirelational_graph_inputs(
    examples,
    model_inputs,
):
    nodes_list = chain_dep_trees(examples["src_dep_trees"])

    connectivity = {}
    for idx, node in enumerate(nodes_list):
        head = node["head"]
        if head == -1:
            continue
        connectivity[idx] = head

    all_dependencies = list(OrderedDict(DEPENDENCIES).keys())

    word_to_subwords = defaultdict(list)
    for idx, word_idx in enumerate(model_inputs.word_ids()):
        if word_idx is None:
            continue
        word_to_subwords[word_idx].append(idx)

    edge_index = []
    edge_type = []
    for from_idx, to_idx in connectivity.items():
        from_indices = word_to_subwords[from_idx]
        to_indices = word_to_subwords[to_idx]

        dep_type = nodes_list[from_idx]["dep"].upper()
        edge_type_idx = all_dependencies.index(dep_type)

        for k_idx in from_indices:
            for v_idx in to_indices:
                edge_index.append([k_idx, v_idx])
                edge_type.append(edge_type_idx)

    # remove last token (eos). May vary depending on the tokenizer
    return {
        "x": model_inputs.input_ids[:-1],
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def create_bipartite_graph_inputs(
    examples,
    model_inputs,
    tokenizer,
    input_is_linearized=False,
):
    def dfs_pre_order(node_list):
        def find_roots():
            roots = []
            for idx, token in enumerate(node_list):
                if token["dep"] == "ROOT":
                    roots.append(idx)
            return roots

        def get_children(node_idx) -> list[int]:
            children = []
            for idx, token in enumerate(node_list):
                if token["head"] == node_idx:
                    children.append(idx)
            return children

        def recursive_dfs(node_idx):
            results = []
            results.append(node_idx)
            for child_idx in get_children(node_idx):
                results += recursive_dfs(child_idx)
            return results

        results = []
        for root_idx in find_roots():
            results += recursive_dfs(root_idx)
        return results

    nodes_list = chain_dep_trees(examples["src_dep_trees"])

    # sanity checks
    # if a graph has a whitespace node, it is malformed
    if any(entry["token"] == " " for entry in nodes_list) or len(nodes_list) <= 1:
        print("Malformed entry.")
        return {
            "x": None,
            "edge_index": None,
        }

    roots = [idx for idx, node in enumerate(nodes_list) if node["dep"] == "ROOT"]

    connectivity = {}
    for idx, node in enumerate(nodes_list):
        head = node["head"]
        if head == -1:
            continue
        connectivity[idx] = head

    word_to_subwords = defaultdict(list)
    for idx, word_idx in enumerate(model_inputs.word_ids()):
        if word_idx is None:
            continue
        word_to_subwords[word_idx].append(idx)

    assert (
        sum(len(value) for value in word_to_subwords.values())
        == len(model_inputs.word_ids()) - 1
    ), "Not all tokens are mapped to subwords."

    if input_is_linearized:
        node_indices = dfs_pre_order(nodes_list)
    else:
        node_indices = list(range(len(nodes_list)))
    nodes_with_padding = [
        [val] if (val in roots) else [f":{nodes_list[val]['dep'].upper()}", val]
        for val in node_indices
    ]
    nodes_with_padding = [item for sublist in nodes_with_padding for item in sublist]

    if not input_is_linearized:
        adjusted_word_to_subwords = defaultdict(list)
        current_token = 0
        for word, subwords in word_to_subwords.items():
            if word not in roots:
                current_token += 1

            adjusted_subwords = [subword + current_token for subword in subwords]
            adjusted_word_to_subwords[word + current_token].extend(adjusted_subwords)

        word_to_subwords = adjusted_word_to_subwords

    try:
        edge_index = []
        for from_idx, to_idx in connectivity.items():
            new_from_idx = nodes_with_padding.index(from_idx)
            new_to_idx = nodes_with_padding.index(to_idx)

            from_indices = word_to_subwords[new_from_idx]
            to_indices = word_to_subwords[new_to_idx]
            relation_idx = from_indices[0] - 1

            for k_idx in from_indices:
                edge_index.append([k_idx, relation_idx])

            for v_idx in to_indices:
                edge_index.append([relation_idx, v_idx])
    except:
        print("Malformed entry.")
        return {
            "x": None,
            "edge_index": None,
        }

    if input_is_linearized:
        # remove last token (eos). May vary depending on the tokenizer
        x = model_inputs.input_ids[:-1]
    else:
        x = []
        for node in nodes_with_padding:
            if isinstance(node, int):
                indices = [
                    i
                    for i, value in enumerate(model_inputs.word_ids())
                    if value == node
                ]
                if indices:
                    x.extend([model_inputs.input_ids[idx] for idx in indices])
            else:
                x.append(tokenizer.convert_tokens_to_ids(node))

    if not edge_index:
        breakpoint()

    if len(x) - 1 != max(set(e for edge in edge_index for e in edge)):
        print("Malformed entry.")
        return {
            "x": None,
            "edge_index": None,
        }

    return {
        "x": x,
        "edge_index": edge_index,
    }


def create_bipartite_graph_from_amr_inputs(
    examples,
    max_source_length,
    tokenizer,
):
    if not examples["node_tokens"]:
        return {
            "x": None,
            "edge_index": None,
        }

    node_inputs = tokenizer(
        examples["node_tokens"],
        max_length=max_source_length,
        padding=False,
        truncation=True,
        add_special_tokens=False,
        is_split_into_words=True,
    )

    word_to_subwords = defaultdict(list)
    for idx, word_idx in enumerate(node_inputs.word_ids()):
        if word_idx is None:
            continue
        word_to_subwords[word_idx].append(idx)

    edge_list = [
        (from_idx, to_idx)
        for (from_idx, to_idx, relation) in examples["edge_triples"]
        if relation == 0
    ]
    edge_index = []
    for from_idx, to_idx in edge_list:
        from_indices = word_to_subwords[from_idx]
        to_indices = word_to_subwords[to_idx]

        for k_idx in from_indices:
            for v_idx in to_indices:
                edge_index.append([k_idx, v_idx])

    if edge_index:
        edge_index_tensor = torch.tensor(
            edge_index
        ).t()  # Transpose to get shape [2, num_edges]
        nodes_with_edges = torch.unique(
            edge_index_tensor.flatten()
        )  # Get unique node indices involved in edges
    else:
        nodes_with_edges = torch.tensor([])

    total_nodes = torch.arange(
        len(node_inputs.input_ids)
    )  # All node indices from 0 to len(x) - 1
    nodes_without_edges = total_nodes[
        ~torch.isin(total_nodes, nodes_with_edges)
    ]  # Find nodes that don't appear in edges
    if len(nodes_without_edges) > 0:
        breakpoint()

    return {
        "x": node_inputs.input_ids,
        "edge_index": edge_index,
    }


def create_multirelational_graph_from_amr_inputs(
    examples,
    max_source_length,
    tokenizer,
):
    if not examples["node_tokens"]:
        return {
            "x": None,
            "edge_index": None,
            "edge_type": None,
        }

    node_inputs = tokenizer(
        examples["node_tokens"],
        max_length=max_source_length,
        padding=False,
        truncation=True,
        add_special_tokens=True,
        is_split_into_words=True,
    )

    word_to_subwords = defaultdict(list)
    for idx, word_idx in enumerate(node_inputs.word_ids()):
        if word_idx is None:
            continue
        word_to_subwords[word_idx].append(idx)

    edge_list = [
        (from_idx, to_idx)
        for (from_idx, to_idx, relation) in examples["edge_triples"]
        if relation == 0
    ]

    edge_index = []
    edge_types = []
    for idx, (from_idx, to_idx) in enumerate(edge_list):
        from_indices = word_to_subwords[from_idx]
        to_indices = word_to_subwords[to_idx]

        for k_idx in from_indices:
            for v_idx in to_indices:
                edge_index.append([k_idx, v_idx])
                edge_types.append(examples["edge_type"][idx])

    return {
        "x": node_inputs.input_ids,
        "edge_index": edge_index,
        "edge_type": edge_types,
    }


def processing_function_for_text_and_AMR(
    tokenizer,
    max_source_length,
    max_target_length,
    graph_representation="multirelational",
    use_linearized_graph=False,
    only_text=False,
):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        # Log the example index around the problem area (e.g., 25% of your dataset size)
        # if 95998 <= idx <= 96000:
        # print(f"Processing example at index {idx}: {examples}")

        # requires examples to be single entry
        assert not isinstance(
            inputs, list
        ), f"You have a batch of {len(inputs)} examples. \
            The linearization of the graph is done instance by instance. \
            Please do not use `batched` examples."

        if use_linearized_graph:
            if not examples["node_tokens"]:
                return {"input_ids": None, "attention_mask": None, "labels": None}
            # inputs = linearize_amr(examples["node_tokens"], examples["edge_triples"])
            inputs = examples["node_tokens"]
            model_inputs = tokenizer(
                inputs,
                max_length=max_source_length,
                padding=False,
                truncation=True,
                is_split_into_words=True,
            )
        else:
            model_inputs = tokenizer(
                inputs,
                max_length=max_source_length,
                padding=False,
                truncation=True,
            )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        if not only_text:
            if graph_representation == "multirelational":
                assert (
                    examples.get("edge_type") is not None
                ), "edge_type missing in example."
                graph_inputs = create_multirelational_graph_from_amr_inputs(
                    examples,
                    max_source_length,
                    tokenizer,
                )
                assert all(
                    key in graph_inputs
                    for key in [
                        "x",
                        "edge_index",
                        "edge_type",
                    ]
                ), "Missing keys in graph_inputs."
            elif graph_representation == "bipartite":
                graph_inputs = create_bipartite_graph_from_amr_inputs(
                    examples,
                    max_source_length,
                    tokenizer,
                )
            else:
                raise ValueError(f"Unknown graph type: {graph_representation}")

            model_inputs = {**model_inputs, **graph_inputs}

            assert all(
                key in model_inputs
                for key in [
                    "input_ids",
                    "labels",
                    "x",
                    "edge_index",
                ]
            ), "Not all items in the list are keys of the dictionary."

            # multirelational graphs should also contain edge_type
            if graph_representation == "multirelational":
                assert "edge_type" in model_inputs, "edge_type missing in model_inputs."
                if not isinstance(examples.get("edge_type"), list):
                    print(f"Error: 'edge_type' is not a list in example: {examples}")
                    raise ValueError("'edge_type' should be a list")
                if not all(isinstance(item, int) for item in examples["edge_type"]):
                    print(
                        f"Error: 'edge_type' contains non-integer values in example: {examples}"
                    )
                    raise ValueError("'edge_type' should only contain integers")

        return model_inputs

    return preprocess_function


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_params_except_adapter(model):
    # Freeze all parameters except self.adapter layers
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def freeze_original_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = any(
            substring in name
            for substring in [
                "shared_cross_attention",
                "adapter",
                "co_attention_layers",
                "graph_cross_attention",
                "decoder",
            ]
        )


def freeze_embeds(model: PreTrainedModel):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def find_double_root_nodes(example):
    for dep_tree in example["src_dep_trees"]:
        if len([node for node in dep_tree if node["dep"] == "ROOT"]) > 1:
            print(example["source"])
            return False
    return True


def chain_dep_trees(dep_trees):
    total_nodes = 0
    chained_dep_tree = []

    for dep_tree in dep_trees:
        adjusted_dep_tree = []

        for node in dep_tree:
            adjusted_node = node.copy()
            if adjusted_node["head"] != -1:
                adjusted_node["head"] += total_nodes
            adjusted_dep_tree.append(adjusted_node)

        total_nodes += len(dep_tree)
        chained_dep_tree.extend(adjusted_dep_tree)

    return chained_dep_tree


def parse_amr_into_reduced_form(example, out_graph_type="bipartite", rel_glossary=None):
    amr = convert_amr_to_graph(
        example["src_amr"], out_graph_type=out_graph_type, rel_glossary=rel_glossary
    )

    if amr is None:
        return {
            **example,
            "node_tokens": None,
            "edge_triples": None,
            "edge_type": None,
        }

    if out_graph_type == "multirelational":
        nodes, triples, edge_type = amr
        return {
            **example,
            "node_tokens": nodes,
            "edge_triples": triples,
            "edge_type": edge_type,
        }
    else:
        nodes, triples = amr
        triples = [
            (from_idx, to_idx, 0 if relation == "d" else 1)
            for from_idx, to_idx, relation in triples
        ]
    return {**example, "node_tokens": nodes, "edge_triples": triples}


def filter_examples_with_missing_edges(examples):
    # Get all node indices from node_tokens
    num_nodes = len(examples["node_tokens"])
    node_indices = set(range(num_nodes))

    # Get all node indices from edge_triples (source and target nodes)
    edge_node_indices = set()
    for from_idx, to_idx, _ in examples["edge_triples"]:
        edge_node_indices.add(from_idx)
        edge_node_indices.add(to_idx)

    # Check if all nodes have edges
    missing_nodes = (
        node_indices - edge_node_indices
    )  # Find nodes that are missing in edge_triples
    if missing_nodes:
        print(f"Filtered out example due to missing nodes: {missing_nodes}")
        return False  # Filter out this example

    return True  # Keep this example