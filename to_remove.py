# %%
import spacy

nlp = spacy.load("en_core_web_lg")


def get_dependency_structure(sentence):
    sent = nlp(sentence)
    src_dp = []
    for token in sent:
        dep_parse = {}
        dep_parse["token"] = token.text
        dep_parse["pos"] = token.tag_
        dep_parse["head"] = token.head.i if token.head.i != token.i else -1
        dep_parse["dep"] = token.dep_
        src_dp.append(dep_parse)
    return src_dp


def add_graphs_to_entries(sample):
    tgt_sents = nlp(sample["source"]).sents
    tgt_dep_trees = []
    for sentence in tgt_sents:
        dep_parse = get_dependency_structure(sentence.text)
        tgt_dep_trees.append(dep_parse)

    sample["src_dep_trees"] = tgt_dep_trees

    return sample


# %%
dep_parse = get_dependency_structure("How can I become a humorous person?")

# %%
from datasets import load_dataset

dataset = load_dataset(
    "processed-data/qqppos",
    data_files={"train": "train_with_dep_tree.jsonl"},
)
dataset = dataset["train"]


# %%
processed_dataset = dataset.map(
    add_graphs_to_entries,
    num_proc=1,
)

# %%
processed_dataset.to_json(
    "processed-data/qqppos/train_with_dep_tree.jsonl", force_ascii=False
)

# %%
# import argparse
# import os.path as osp
# import time

# import torch
# import torch.nn.functional as F
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import log
# from torch_geometric.nn import GCNConv

# args = argparse.Namespace()
# args.dataset = "Cora"
# args.hidden_channels = 16
# args.lr = 0.01
# args.epochs = 200

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")


# path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
# dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
# data = dataset[0].to(device)


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
#         self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.dropout(x, p=0.5, training=self.training)
#         print(x.shape)
#         x = self.conv1(x, edge_index, edge_weight).relu()
#         print(x.shape)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return x


# model = GCN(
#     in_channels=dataset.num_features,
#     hidden_channels=args.hidden_channels,
#     out_channels=dataset.num_classes,
# ).to(device)

# optimizer = torch.optim.Adam(
#     [
#         dict(params=model.conv1.parameters(), weight_decay=5e-4),
#         dict(params=model.conv2.parameters(), weight_decay=0),
#     ],
#     lr=args.lr,
# )  # Only perform weight-decay on first convolution.


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_attr)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
#     return accs


# # %%
# best_val_acc = test_acc = 0
# times = []
# for epoch in range(1, args.epochs + 1):
#     start = time.time()
#     loss = train()
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
#     times.append(time.time() - start)
# print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# # %%
# tokenized_tokens = [
#     [37],
#     [14512],
#     [774],
#     [13],
#     [16312],
#     [1636],
#     [3, 3959],
#     [47],
#     [8],
#     [604, 189],
#     [774],
#     [13],
#     [8],
#     [868],
#     [21249],
#     [2125],
#     [3, 5],
# ]

# nodes = [
#     {"dep": "det", "head": 2, "pos": "DT", "token": "The"},
#     {"dep": "compound", "head": 2, "pos": "NNP", "token": "NBA"},
#     {"dep": "nsubj", "head": 7, "pos": "NN", "token": "season"},
#     {"dep": "prep", "head": 2, "pos": "IN", "token": "of"},
#     {"dep": "pobj", "head": 3, "pos": "CD", "token": "1975"},
#     {"dep": "punct", "head": 7, "pos": ":", "token": "--"},
#     {"dep": "nsubj", "head": 7, "pos": "CD", "token": "76"},
#     {"dep": "ROOT", "head": -1, "pos": "VBD", "token": "was"},
#     {"dep": "det", "head": 10, "pos": "DT", "token": "the"},
#     {"dep": "amod", "head": 10, "pos": "JJ", "token": "30th"},
#     {"dep": "attr", "head": 7, "pos": "NN", "token": "season"},
#     {"dep": "prep", "head": 10, "pos": "IN", "token": "of"},
#     {"dep": "det", "head": 15, "pos": "DT", "token": "the"},
#     {"dep": "compound", "head": 15, "pos": "NNP", "token": "National"},
#     {"dep": "compound", "head": 15, "pos": "NNP", "token": "Basketball"},
#     {"dep": "pobj", "head": 11, "pos": "NNP", "token": "Association"},
#     {"dep": "punct", "head": 7, "pos": ".", "token": "."},
# ]

# # %%

# from collections import OrderedDict

# from glossary import DEPENDENCIES


# def get_index(tokenized_tokens, n_idx, current_idx):
#     index = 0
#     for i in range(n_idx):
#         index += len(tokenized_tokens[i])
#     index += current_idx
#     return index


# graph_representation = "bipartite"

# x = []
# edge_index = []
# edge_type = []
# node_types_count = 0
# new_node_types = []
# all_dependencies = list(OrderedDict(DEPENDENCIES).keys())
# total_tokens = sum(len(sublist) for sublist in tokenized_tokens)
# for n_idx, node in enumerate(nodes):
#     x += tokenized_tokens[n_idx]

#     head = node["head"]
#     if head == -1:
#         continue

#     for from_idx in range(len(tokenized_tokens[n_idx])):
#         for to_idx in range(len(tokenized_tokens[head])):
#             if graph_representation == "regular":
#                 edge_index.append(
#                     [
#                         get_index(tokenized_tokens, n_idx, from_idx),
#                         get_index(tokenized_tokens, head, to_idx),
#                     ]
#                 )
#                 edge_type.append(all_dependencies.index(node["dep"].upper()))
#             elif graph_representation == "bipartite":
#                 relation_idx = total_tokens + node_types_count
#                 print(n_idx, head, from_idx, to_idx, relation_idx)
#                 edge_index.append(
#                     [
#                         get_index(tokenized_tokens, n_idx, from_idx),
#                         relation_idx,
#                     ]
#                 )
#                 edge_index.append(
#                     [
#                         relation_idx,
#                         get_index(tokenized_tokens, head, to_idx),
#                     ]
#                 )

#             else:
#                 raise ValueError(f"Unknown graph type: {graph_representation}")
#     node_types_count += 1
#     new_node_types.append(relation_idx)

# print(x)
# print(edge_index)


# # %%
# import spacy

# nlp = spacy.load("en_core_web_lg")
# # %%
# t = nlp("Sally is understanding.")
# # %%
# for i in t:
#     print(i.text, i.dep_, i.head)
# # %%

# %%
import torch

# Example edge_index_batches
edge_index_batches = [
    torch.tensor(
        [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 4, 4, 1, 4, 5, 9, 9, 6, 1, 1]]
    ),
    torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8, 9], [4, 4, 4, 4, 4, 4, 6, 6, 6]]),
]
batch = torch.cat(
    [
        torch.full((edge_index_batch.size(1),), i)
        for i, edge_index_batch in enumerate(edge_index_batches)
    ]
)

# Display batch tensor
print("Batch Tensor:")
print(batch)
# %%
from collections import defaultdict

# Example usage:
nodes_list = [
    {"dep": "det", "head": 2, "pos": "DT", "token": "The"},
    {"dep": "compound", "head": 2, "pos": "NNP", "token": "NBA"},
    {"dep": "nsubj", "head": 7, "pos": "NN", "token": "season"},
    {"dep": "prep", "head": 2, "pos": "IN", "token": "of"},
    {"dep": "pobj", "head": 3, "pos": "CD", "token": "1975"},
    {"dep": "punct", "head": 7, "pos": ":", "token": "--"},
    {"dep": "nsubj", "head": 7, "pos": "CD", "token": "76"},
    {"dep": "ROOT", "head": -1, "pos": "VBD", "token": "was"},
    {"dep": "det", "head": 10, "pos": "DT", "token": "the"},
    {"dep": "amod", "head": 10, "pos": "JJ", "token": "30th"},
    {"dep": "attr", "head": 7, "pos": "NN", "token": "season"},
    {"dep": "prep", "head": 10, "pos": "IN", "token": "of"},
    {"dep": "det", "head": 15, "pos": "DT", "token": "the"},
    {"dep": "compound", "head": 15, "pos": "NNP", "token": "National"},
    {"dep": "compound", "head": 15, "pos": "NNP", "token": "Basketball"},
    {"dep": "pobj", "head": 11, "pos": "NNP", "token": "Association"},
    {"dep": "punct", "head": 7, "pos": ".", "token": "."},
]
word_ids = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    32,
    None,
]

# %%
nodes_list = [
    {"dep": "det", "head": 1, "token": "The"},
    {"dep": "compound", "head": 2, "token": "NBA"},
    {"dep": "ROOT", "head": -1, "token": "was"},
    # {"dep": "prep", "head": 2, "token": "of"},
]
word_ids = [
    0,
    1,
    1,
    2,
    2,
    # 3,
    # 3,
    # 3,
    None,
]


# %%
def dfs_pre_order(node_list):
    def find_root():
        for idx, token in enumerate(node_list):
            if token["dep"] == "ROOT":
                return idx
        return None

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

    return recursive_dfs(find_root())


dfs_ordered_nodes = dfs_pre_order(nodes_list)

# %%
connectivity = {}
for idx, node in enumerate(nodes_list):
    head = node["head"]
    if head == -1:
        continue
    connectivity[idx] = head

# %%
from collections import defaultdict

word_to_subwords = defaultdict(list)
for idx, word_idx in enumerate(word_ids):
    if word_idx is None:
        continue
    word_to_subwords[word_idx].append(idx)

# %%
ordered_nodes_with_padding = [
    [val] if idx == 0 else [-1, val] for idx, val in enumerate(dfs_ordered_nodes)
]
ordered_nodes_with_padding = [
    item for sublist in ordered_nodes_with_padding for item in sublist
]

# %%
need_adjustment = False
if need_adjustment:
    adjusted_word_to_subwords = defaultdict(list)
    current_token = 0
    for word, subwords in word_to_subwords.items():
        if word != 0:
            current_token += 1

        adjusted_subwords = [subword + current_token for subword in subwords]
        adjusted_word_to_subwords[word + current_token].extend(adjusted_subwords)

    word_to_subwords = adjusted_word_to_subwords

# %%
edge_index = []

for from_idx, to_idx in connectivity.items():
    new_from_idx = ordered_nodes_with_padding.index(from_idx)
    new_to_idx = ordered_nodes_with_padding.index(to_idx)

    from_indices = word_to_subwords[new_from_idx]
    to_indices = word_to_subwords[new_to_idx]
    relation_idx = from_indices[0] - 1

    for k_idx in from_indices:
        edge_index.append([k_idx, relation_idx])

    for v_idx in to_indices:
        edge_index.append([relation_idx, v_idx])

print(edge_index)
# %%
nodes_list = [
    {"dep": "det", "head": 3, "pos": "DT", "token": "The"},
    {"dep": "npadvmod", "head": 0, "pos": "CD", "token": "2007"},
    {"dep": "punct", "head": 0, "pos": ":", "token": "--"},
    {"dep": "ROOT", "head": -1, "pos": "CD", "token": "08"},
    {"dep": "compound", "head": 1, "pos": "NNP", "token": "Kansas"},
    {"dep": "compound", "head": 3, "pos": "NNP", "token": "State"},
    {"dep": "compound", "head": 3, "pos": "NNPS", "token": "Wildcats"},
    {"dep": "poss", "head": 6, "pos": "NNPS", "token": "Men"},
    {"dep": "case", "head": 3, "pos": "POS", "token": "'s"},
    {"dep": "compound", "head": 6, "pos": "NNP", "token": "Basketball"},
    {"dep": "nsubj", "head": 7, "pos": "NNP", "token": "Team"},
    {"dep": "ROOT", "head": -1, "pos": "VBZ", "token": "represents"},
    {"dep": "compound", "head": 9, "pos": "NNP", "token": "Kansas"},
    {"dep": "compound", "head": 10, "pos": "NNP", "token": "State"},
    {"dep": "dobj", "head": 7, "pos": "NNP", "token": "University"},
    {"dep": "prep", "head": 7, "pos": "IN", "token": "at"},
    {"dep": "det", "head": 20, "pos": "DT", "token": "the"},
    {"dep": "nummod", "head": 20, "pos": "CD", "token": "2007"},
    {"dep": "punct", "head": 15, "pos": ":", "token": "--"},
    {"dep": "nummod", "head": 20, "pos": "CD", "token": "08"},
    {"dep": "compound", "head": 18, "pos": "NNP", "token": "College"},
    {"dep": "punct", "head": 18, "pos": "HYPH", "token": "-"},
    {"dep": "compound", "head": 20, "pos": "NNP", "token": "Basketball"},
    {"dep": "punct", "head": 20, "pos": "HYPH", "token": "-"},
    {"dep": "pobj", "head": 11, "pos": "NNP", "token": "Season"},
    {"dep": "punct", "head": 7, "pos": ".", "token": "."},
]

# %%
from dependency_tree import TreeNode

tree1 = TreeNode.from_parse_trees([nodes_list])

# %%
s = "The 2007 -- 08 Kansas State Wildcats Men 's Basketball Team represents Kansas State University at the 2007 -- 08 College - Basketball - Season ."

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_lg")
doc = nlp(s)
displacy.serve(doc, style="dep")


# %%
from collections import defaultdict

from utils import chain_dep_trees

nodes_list = [
    {"dep": "det", "head": 3, "pos": "DT", "token": "The"},
    {"dep": "npadvmod", "head": 0, "pos": "CD", "token": "2007"},
    {"dep": "punct", "head": 0, "pos": ":", "token": "--"},
    {"dep": "ROOT", "head": -1, "pos": "CD", "token": "08"},
    {"dep": "compound", "head": 5, "pos": "NNP", "token": "Kansas"},
    {"dep": "compound", "head": 7, "pos": "NNP", "token": "State"},
    {"dep": "compound", "head": 7, "pos": "NNPS", "token": "Wildcats"},
    {"dep": "poss", "head": 10, "pos": "NNPS", "token": "Men"},
    {"dep": "case", "head": 7, "pos": "POS", "token": "'s"},
    {"dep": "compound", "head": 10, "pos": "NNP", "token": "Basketball"},
    {"dep": "nsubj", "head": 11, "pos": "NNP", "token": "Team"},
    {"dep": "ROOT", "head": -1, "pos": "VBZ", "token": "represents"},
    {"dep": "compound", "head": 13, "pos": "NNP", "token": "Kansas"},
    {"dep": "compound", "head": 14, "pos": "NNP", "token": "State"},
    {"dep": "dobj", "head": 11, "pos": "NNP", "token": "University"},
    {"dep": "prep", "head": 11, "pos": "IN", "token": "at"},
    {"dep": "det", "head": 24, "pos": "DT", "token": "the"},
    {"dep": "nummod", "head": 24, "pos": "CD", "token": "2007"},
    {"dep": "punct", "head": 19, "pos": ":", "token": "--"},
    {"dep": "nummod", "head": 24, "pos": "CD", "token": "08"},
    {"dep": "compound", "head": 22, "pos": "NNP", "token": "College"},
    {"dep": "punct", "head": 22, "pos": "HYPH", "token": "-"},
    {"dep": "compound", "head": 24, "pos": "NNP", "token": "Basketball"},
    {"dep": "punct", "head": 24, "pos": "HYPH", "token": "-"},
    {"dep": "pobj", "head": 15, "pos": "NNP", "token": "Season"},
    {"dep": "punct", "head": 11, "pos": ".", "token": "."},
]

word_ids = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    16,
    16,
    17,
    18,
    18,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    44,
    45,
    46,
    46,
    47,
    48,
    48,
    None,
]

detokenized_input_ids = [
    "▁08",
    ":DET",
    "▁The",
    ":NPADVMOD",
    "▁2007",
    ":PUNCT",
    "▁--",
    "re",
    "present",
    "s",
    ":NSUBJ",
    "▁Team",
    ":POSS",
    "▁Men",
    ":COMPOUND",
    "▁State",
    ":COMPOUND",
    "▁Kansas",
    ":COMPOUND",
    "▁Wild",
    "cat",
    "s",
    ":CASE",
    "▁",
    "'",
    "s",
    ":COMPOUND",
    "▁Basketball",
    ":DOBJ",
    "▁University",
    ":COMPOUND",
    "▁State",
    ":COMPOUND",
    "▁Kansas",
    ":PREP",
    "▁at",
    ":POBJ",
    "▁Season",
    ":DET",
    "▁the",
    ":NUMMOD",
    "▁2007",
    ":NUMMOD",
    "▁08",
    ":PUNCT",
    "▁--",
    ":COMPOUND",
    "▁Basketball",
    ":COMPOUND",
    "▁College",
    ":PUNCT",
    "▁",
    "-",
    ":PUNCT",
    "▁",
    "-",
    ":PUNCT",
    "▁",
    ".",
    "</s>",
]

# nodes_list = [
#     {"dep": "det", "head": 2, "pos": "DT", "token": "The"},
#     {"dep": "npadvmod", "head": 0, "pos": "CD", "token": "2007"},
#     {"dep": "ROOT", "head": -1, "pos": "CD", "token": "08"},
#     {"dep": "npadvmod", "head": 4, "pos": "CD", "token": "q"},
#     {"dep": "ROOT", "head": -1, "pos": "CD", "token": "a"},
# ]

# word_ids = [
#     0,
#     1,
#     2,
#     3,
#     4,
#     4,
#     5,
#     6,
#     6,
#     6,
#     7,
#     8,
#     None,
# ]

roots = [3, 11]

need_adjustment = False


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


# nodes_list = chain_dep_trees(examples["src_dep_trees"])
dfs_ordered_nodes = dfs_pre_order(nodes_list)

connectivity = {}
for idx, node in enumerate(nodes_list):
    head = node["head"]
    if head == -1:
        continue
    connectivity[idx] = head

word_to_subwords = defaultdict(list)
for idx, word_idx in enumerate(word_ids):  # model_inputs.word_ids()):
    if word_idx is None:
        continue
    word_to_subwords[word_idx].append(idx)

ordered_nodes_with_padding = [
    [val] if (val in roots) else [-1, val] for val in dfs_ordered_nodes
]
ordered_nodes_with_padding = [
    item for sublist in ordered_nodes_with_padding for item in sublist
]

if need_adjustment:
    adjusted_word_to_subwords = defaultdict(list)
    current_token = 0
    for word, subwords in word_to_subwords.items():
        if word != 0:
            current_token += 1

        adjusted_subwords = [subword + current_token for subword in subwords]
        adjusted_word_to_subwords[word + current_token].extend(adjusted_subwords)

    word_to_subwords = adjusted_word_to_subwords

# %%
edge_index = []

for from_idx, to_idx in connectivity.items():
    print(f"from_idx: {from_idx}")
    print(f"to_idx: {to_idx}")
    new_from_idx = ordered_nodes_with_padding.index(from_idx)
    new_to_idx = ordered_nodes_with_padding.index(to_idx)

    from_indices = word_to_subwords[new_from_idx]
    to_indices = word_to_subwords[new_to_idx]
    relation_idx = from_indices[0] - 1

    print(f"from_indices: {from_indices}")
    print(f"to_indices: {to_indices}")
    print(f"relation_idx: {relation_idx}")

    for k_idx in from_indices:
        edge_index.append([k_idx, relation_idx])

    for v_idx in to_indices:
        edge_index.append([relation_idx, v_idx])

# %%
words = [
    detokenized_input_ids[idx]
    for idx in range(len(word_ids))
    if word_ids[idx] is not None and word_ids[idx] % 2 == 0
]

relations = [
    detokenized_input_ids[idx]
    for idx in range(len(word_ids))
    if word_ids[idx] is not None and word_ids[idx] % 2 == 1
]

print(words)
print(relations)


# %%
def chain_dep_trees(dep_trees):
    total_nodes = 0

    chained_dep_tree = []

    for dep_tree in dep_trees:
        adjusted_dep_tree = []

        for node in dep_tree:
            adjusted_node = node.copy()
            adjusted_node["head"] += total_nodes
            adjusted_dep_tree.append(adjusted_node)

        total_nodes += len(dep_tree)
        chained_dep_tree.extend(adjusted_dep_tree)

    return chained_dep_tree


dep_tree = [
    [
        {"dep": "det", "head": 3, "pos": "DT", "token": "The"},
        {"dep": "npadvmod", "head": 0, "pos": "CD", "token": "2007"},
        {"dep": "punct", "head": 0, "pos": ":", "token": "--"},
        {"dep": "ROOT", "head": -1, "pos": "CD", "token": "08"},
    ],
    [
        {"dep": "compound", "head": 1, "pos": "NNP", "token": "Kansas"},
        {"dep": "compound", "head": 3, "pos": "NNP", "token": "State"},
        {"dep": "compound", "head": 3, "pos": "NNPS", "token": "Wildcats"},
        {"dep": "poss", "head": 6, "pos": "NNPS", "token": "Men"},
        {"dep": "case", "head": 3, "pos": "POS", "token": "'s"},
        {"dep": "compound", "head": 6, "pos": "NNP", "token": "Basketball"},
        {"dep": "nsubj", "head": 7, "pos": "NNP", "token": "Team"},
        {"dep": "ROOT", "head": -1, "pos": "VBZ", "token": "represents"},
        {"dep": "compound", "head": 9, "pos": "NNP", "token": "Kansas"},
        {"dep": "compound", "head": 10, "pos": "NNP", "token": "State"},
        {"dep": "dobj", "head": 7, "pos": "NNP", "token": "University"},
        {"dep": "prep", "head": 7, "pos": "IN", "token": "at"},
        {"dep": "det", "head": 20, "pos": "DT", "token": "the"},
        {"dep": "nummod", "head": 20, "pos": "CD", "token": "2007"},
        {"dep": "punct", "head": 15, "pos": ":", "token": "--"},
        {"dep": "nummod", "head": 20, "pos": "CD", "token": "08"},
        {"dep": "compound", "head": 18, "pos": "NNP", "token": "College"},
        {"dep": "punct", "head": 18, "pos": "HYPH", "token": "-"},
        {"dep": "compound", "head": 20, "pos": "NNP", "token": "Basketball"},
        {"dep": "punct", "head": 20, "pos": "HYPH", "token": "-"},
        {"dep": "pobj", "head": 11, "pos": "NNP", "token": "Season"},
        {"dep": "punct", "head": 7, "pos": ".", "token": "."},
    ],
]

# Example usage
chained_dep_tree = chain_dep_trees(dep_tree)
# %%
nodes_with_padding = [
    -1,
    0,
    -1,
    1,
    -1,
    2,
    -1,
    3,
    -1,
    4,
    -1,
    5,
    -1,
    6,
    7,
    -1,
    8,
    -1,
    9,
    -1,
    10,
    -1,
    11,
    -1,
    12,
    -1,
    13,
    -1,
    14,
    -1,
    15,
    -1,
    16,
]

word_to_subwords = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6, 7],
    7: [8],
    8: [9],
    9: [10, 11],
    10: [12],
    11: [13],
    12: [14],
    13: [15],
    14: [16],
    15: [17],
    16: [18, 19],
}

roots = [7]
# %%
from collections import defaultdict

adjusted_word_to_subwords = defaultdict(list)
current_token = 0
for word, subwords in word_to_subwords.items():
    if word not in roots:
        current_token += 1

    adjusted_subwords = [subword + current_token for subword in subwords]
    adjusted_word_to_subwords[word + current_token].extend(adjusted_subwords)

# %%

# %%
sentences = [
    "the alcohol and cigarettes are banned , right ?",
    "are n't alcohol and cigarettes banned ?",
    "he got a cab ? how was that done ?",
    "i do n't throw things . . . i teach .",
]


# %%
import re

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()

for sentence in sentences:
    # convert ". . ." into "..."
    sentence = re.sub(r"\.\s*\.\s*\.", "...", sentence)
    normalized_sentence = detokenizer.detokenize(sentence.split())
# %%
s1 = "It is situated south of Köroğlu Mountains and to the north of Bolu ."
t1 = "It is situated south of Körolu - mountains and north of the Bolu."
p1 = "It is situated south of Köroğlu Mountains and to the north of Bolu ."

s2 = "It is situated south of Körolu Mountains and to the north of Bolu."
t2 = "It is situated south of Körolu - mountains and north of the Bolu."
p2 = "It is situated south of Körolu Mountains and to the north of Bolu."

# %%
import pandas as pd
from preprocessing_utils import clean_sentence

# file_path = "output/t5-small-paranmt-small-1e-4-100e-concat-bipartite-first-512-no_freeze/eval_generations.csv"
# file_path = "output/t5-small-paranmt-small-1e-4-100e-linearized/eval_generations.csv"
file_path = "output/t5-small-paranmt-small-standard/eval_generations.csv"

dataset = pd.read_csv(file_path, sep="\t")
# %%
cleaned_dataset = dataset.applymap(clean_sentence)
# %%
cleaned_dataset.to_csv(
    file_path,
    index=False,
    sep="\t",
)


# %%
def linearize_graph(node_tokens, edge_triples):
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

    return " ".join(linearized_nodes)


# Given graph
graph_data = {
    "node_tokens": ["sorry-01", ":polarity", "amr-unknown", ":ARG1", "i"],
    "edge_triples": [
        [0, 1, 0],
        [0, 3, 0],
        [1, 0, 1],
        [1, 2, 0],
        [2, 1, 1],
        [3, 0, 1],
        [3, 4, 0],
        [4, 3, 1],
    ],
}

linearized_result = linearize_graph(
    graph_data["node_tokens"], graph_data["edge_triples"]
)

# Print the linearized result
print("Linearized Result:", linearized_result)

# %%
from transformers import AutoTokenizer

from utils import create_bipartite_graph_from_amr_inputs

examples = {
    "node_tokens": [
        "put-01",
        ":mode",
        "imperative",
        ":ARG0",
        "we",
        ":ARG1",
        "something",
        ":ARG1-of",
        "cold-01",
        ":ARG2",
        "there",
    ],
    "edge_triples": [
        [0, 1, 0],
        [0, 3, 0],
        [0, 5, 0],
        [0, 9, 0],
        [1, 0, 1],
        [1, 2, 0],
        [2, 1, 1],
        [3, 0, 1],
        [3, 4, 0],
        [4, 3, 1],
        [5, 0, 1],
        [5, 6, 0],
        [6, 5, 1],
        [6, 7, 0],
        [7, 6, 1],
        [7, 8, 0],
        [8, 7, 1],
        [9, 0, 1],
        [9, 10, 0],
        [10, 9, 1],
    ],
}
tokenizer = AutoTokenizer.from_pretrained(
    "t5-large",
    extra_ids=0,  # no need for sentinel tokens
    additional_special_tokens=[":mode", ":ARG0", ":ARG1", ":ARG1-of", ":ARG2"],
    use_fast=True,
)
graph_inputs = create_bipartite_graph_from_amr_inputs(
    examples,
    256,
    tokenizer,
)
# %%
