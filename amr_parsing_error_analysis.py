# %%
import os

from datasets import load_dataset

dataset_name = os.path.join("processed-data", "qqppos")  # "qqppos"  # paranmt-small

# %%
# Load dataset
raw_datasets = load_dataset(
    dataset_name,
    data_files={
        "train": f"train_with_amr.jsonl",
        "validation": f"validation_with_amr.jsonl",
        "test": f"test_with_amr.jsonl",
    },
)

# %%
# Process and save dataset with reduced AMR graphs
from utils import parse_amr_into_reduced_form

raw_datasets = raw_datasets.map(
    parse_amr_into_reduced_form,
    batched=False,
    load_from_cache_file=False,
    desc="Parsing AMR into reduced form",
)


# %%
def save_dataset_to_jsonl(dataset, filepath):
    df = dataset.to_pandas()
    df.to_json(filepath, orient="records", lines=True)


for split in raw_datasets:
    save_dataset_to_jsonl(
        raw_datasets[split],
        f"{dataset_name}/{split}_reduced_amr.jsonl",
    )

# %%
raw_datasets = load_dataset(
    dataset_name,
    data_files={
        "train": f"train_reduced_amr.jsonl",
        "validation": f"validation_reduced_amr.jsonl",
        "test": f"test_reduced_amr.jsonl",
    },
)

# %%
# Add index to each example
raw_datasets = raw_datasets.map(lambda example, idx: {"index": idx}, with_indices=True)

# %%
# Initialize lists to keep track of removed indices
removed_indices_cleaning = []
removed_indices_edge_triples = []


# Clean malformed entries and track removed indices
def filter_malformed(example):
    if all(value is not None for value in example.values()):
        return True
    removed_indices_cleaning.append(example["index"])
    return False


clean_datasets = raw_datasets.filter(filter_malformed)


# Filter edge_triples and track removed indices
def filter_edge_triples(example):
    example["edge_triples"] = [
        entry for entry in example["edge_triples"] if entry[2] != 1
    ]
    if example["edge_triples"] is not None and len(example["edge_triples"]) > 0:
        return True
    removed_indices_edge_triples.append(example["index"])
    return False


clean_datasets = clean_datasets.filter(filter_edge_triples)

# Print removed indices for inspection
print("Indices removed during cleaning:", removed_indices_cleaning)
print("Indices removed during edge_triples filtering:", removed_indices_edge_triples)


# %%
def save_dataset_to_jsonl(dataset, filepath):
    df = dataset.to_pandas()
    df.to_json(filepath, orient="records", lines=True)


for split in clean_datasets:
    save_dataset_to_jsonl(
        clean_datasets[split],
        f"{dataset_name}/{split}_clean_with_amr_and_codes.jsonl",
    )

# %%
import nltk

count_edge_triples_none = 0
total_source_tokens_none = 0

count_edge_triples_empty = 0
total_source_tokens_empty = 0

# Iterate through the training set and count entries
for example in raw_datasets["train"]:
    source_tokens = nltk.word_tokenize(example["source"])

    if example["edge_triples"] is None:
        count_edge_triples_none += 1
        total_source_tokens_none += len(source_tokens)

    if example["edge_triples"] == []:
        count_edge_triples_empty += 1
        total_source_tokens_empty += len(source_tokens)

# Calculate the mean number of tokens
mean_source_tokens_none = (
    total_source_tokens_none / count_edge_triples_none
    if count_edge_triples_none > 0
    else 0
)
mean_source_tokens_empty = (
    total_source_tokens_empty / count_edge_triples_empty
    if count_edge_triples_empty > 0
    else 0
)

# Print the results
print(f"Total count of entries with edge_triples as None: {count_edge_triples_none}")
print(
    f"Mean number of tokens in 'source' sentences of entries with edge_triples as None: {mean_source_tokens_none:.2f}"
)

print(
    f"Total count of entries with edge_triples as empty list: {count_edge_triples_empty}"
)
print(
    f"Mean number of tokens in 'source' sentences of entries with edge_triples as empty list: {mean_source_tokens_empty:.2f}"
)


# %%
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Download the NLTK tokenizer data
nltk.download("punkt")

combined_dataset = pd.concat(
    [
        raw_datasets["train"].to_pandas(),
        raw_datasets["validation"].to_pandas(),
        raw_datasets["test"].to_pandas(),
    ]
)


# %%
# Function to compute the mean number of tokens
def compute_mean_tokens(dataset, key="source"):
    total_tokens = 0
    total_entries = len(dataset)

    for _, row in dataset.iterrows():
        tokens = word_tokenize(row[key])
        total_tokens += len(tokens)

    mean_tokens = total_tokens / total_entries

    return mean_tokens


# %%
# Compute the mean number of tokens for the combined dataset
mean_source_tokens = compute_mean_tokens(combined_dataset)

print(f"Mean Source Tokens: {mean_source_tokens}")


# %%

# Find missing indices in dataset
import json
import os

from datasets import load_dataset

dataset_name = "paranmt-small"  # "qqppos"  # paranmt-small
dataset_path = os.path.join("processed-data", dataset_name)

processed_datasets = load_dataset(
    dataset_path,
    data_files={
        "train": f"train_clean_with_amr_and_codes.jsonl",
        "validation": f"validation_clean_with_amr_and_codes.jsonl",
        "test": f"test_clean_with_amr_and_codes.jsonl",
    },
)

missing_indices = {}
for split_name, split_data in processed_datasets.items():
    indices = split_data["index"]

    max_index = max(split_data["index"])
    expected_indices = set(range(max_index + 1))

    actual_indices = set(indices)

    missing = expected_indices - actual_indices

    missing_indices[split_name] = sorted(missing)

with open(f"missing_indices_{dataset_name}.json", "w") as json_file:
    json.dump(missing_indices, json_file, indent=4)

# %%
