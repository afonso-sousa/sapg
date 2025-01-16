from collections import defaultdict


def _find_roots(tree):
    roots = []
    for idx, token in enumerate(tree):
        if token["dep"] == "ROOT":
            roots.append(idx)
    return roots


def _get_children(tree, node_idx) -> list[int]:
    children = []
    for idx, token in enumerate(tree):
        if token["head"] == node_idx:
            children.append(idx)
    return children


def _traverse(tree, current_node_idx, visited):
    result = []
    # Check if the node has been visited
    if current_node_idx in visited:
        return result

    # Add the current node to the visited list
    visited.append(current_node_idx)

    children = _get_children(tree, current_node_idx)
    relation = tree[current_node_idx]["dep"].upper()
    relation_str = f":{relation} " if tree[current_node_idx]["head"] != -1 else ""

    children_output = []
    for child in children:
        children_output += [_traverse(tree, child, visited)]

    children_output = " ".join(children_output)
    children_output = " " + children_output if children_output else ""

    result = f"{relation_str}{tree[current_node_idx]['token']}" + children_output

    return result


def linearize(all_nodes):
    result = ""
    # all_nodes = chain_dep_trees(dep_parse_list)
    for root_idx in _find_roots(all_nodes):
        output = _traverse(all_nodes, root_idx, [])
        result = result + " " + output

    return result


def linearize_amr(node_tokens, edge_triples):
    def topological_sort(node, visited, stack):
        visited[node] = True
        if node in graph:
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    topological_sort(neighbor, visited, stack)
        stack.append(node)

    graph = defaultdict(list)
    for edge in edge_triples:
        if edge[2] == 0:  # Only consider edges with the last index equal to 0
            graph[edge[0]].append(edge[1])

    visited = {i: False for i in range(len(node_tokens))}
    stack = []

    for node in range(len(node_tokens)):
        if not visited[node]:
            topological_sort(node, visited, stack)

    linearized_nodes = [node_tokens[i] for i in reversed(stack)]

    return linearized_nodes
