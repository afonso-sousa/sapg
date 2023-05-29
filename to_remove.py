# %%
from datasets import load_dataset

d = load_dataset("processed-data/paws",
    data_files={
        "train": "train_with_reduced_graph.json",
        "validation": "validation_with_reduced_graph.json",
        "test": "test_with_reduced_graph.json",
    }
)
# %%
d = d.remove_columns(["nodes", "edges"])

# %%
for dataset_name, dataset in d.items():
    dataset.to_json(f"data/{dataset_name}.jsonl", orient="records", lines=True)

#########################################################

# %%
from datasets import load_dataset

d = load_dataset("data/paws")

# %%
sample = d["train"][0]

# %%
import itertools

all_nodes = list(itertools.chain.from_iterable(sample["dp"]))
all_dependencies = list(set(node["dep"] for node in all_nodes))
num_relations = len(all_dependencies)

# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
input_ids = tokenizer(sample["source"], return_tensors="pt").input_ids
labels = tokenizer(sample["target"], return_tensors="pt").input_ids

# %%
from transformers import AutoConfig

from graph_t5 import GraphT5ForConditionalGeneration

config = AutoConfig.from_pretrained("t5-small")
config.adapter_dim = 1024
config.num_relations = num_relations

# %%
model2 = GraphT5ForConditionalGeneration.from_pretrained("t5-small", config=config)

# %%
edges = []
edge_type = []
for i, node in enumerate(all_nodes):
    if node["head"] == -1:
        continue
    edges.append(
        [
            i,
            node["head"],
        ]
    )
    edge_type.append(all_dependencies.index(node["dep"]))


# %%
import torch
from torch_geometric.data import Batch, Data

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_type = torch.tensor(edge_type)
graph = Data(edge_index=edge_index, edge_type=edge_type)
graph_batch = Batch.from_data_list([graph])
# %%
loss = model2(input_ids=input_ids, labels=labels, graphs=graph_batch).loss

loss.item()

# %%
# def get_mean_embeddings(self, input_ids, attention_mask):
#     bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
#     attention_mask = attention_mask.unsqueeze(-1)
#     mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
#     return mean_output

nodes = [node["word"] for node in source_graph["nodes"]]

node_features = tokenizer(nodes, max_length=12, padding="max_length", truncation=True)
tensor_nodes = torch.tensor(node_features["input_ids"])
# emb_weights = model.shared.weight  # 32128, 768 => vocab_size, emb_size
# get_mean_embeddings(node_features, model.shared)
node_feats = model.shared(tensor_nodes).mean(1)

# %%
edges = []
edge_type = []
for i, edge_vec in enumerate(source_graph["edges"]):
    for j, relation in enumerate(edge_vec):
        if relation != "" and relation != "SELF":
            edges.append(
                [
                    i,
                    j,
                ]
            )
            edge_type.append(all_dependencies.index(relation))

# %%
x = node_feats.type(torch.float)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_type = torch.tensor(edge_type)
graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)

# %%
dloader = PyGDataLoader(
    [{"input_ids": sample["source"], "label": sample["target"], "graph": graph}],
    batch_size=1,
)

# %%
from transformers import T5Tokenizer

from graph_collator import GraphDataCollatorForSeq2Seq

tokenizer = T5Tokenizer.from_pretrained("t5-small")


examples = [
    {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [1, 2, 3],
        "edge_index": [[1, 0], [2, 1]],
        "edge_type": [1, 0],
    },
    {
        "input_ids": [4, 5, 6],
        "attention_mask": [1, 1, 1],
        "labels": [4, 5, 6],
        "edge_index": [[1, 2], [2, 0]],
        "edge_type": [1, 0],
    },
]

# Instantiate the data collator
data_collator = GraphDataCollatorForSeq2Seq(tokenizer)

# Call the data collator on the examples
collated_data = data_collator(examples)
# %%
import torch
from torch_geometric.data import Batch, Data

# Create a list of Data objects
graphs = [
    Data(edge_index=torch.tensor([[0, 2], [1, 2], [2, 10], [3, 2], [4, 6], [5, 6], [6, 3], [7, 6], [8, 7], [9, 10], [11, 12], [12, 10], [13, 10]])),
    Data(edge_index=torch.tensor([[0, 2], [1, 2], [3, 2], [4, 3], [5, 4], [6, 4], [7, 2], [8, 2], [9, 8], [10, 9], [11, 10], [13, 2]])),
]

# Create a batch of graphs using Batch.from_data_list
batch = Batch.from_data_list(graphs)
# %%
