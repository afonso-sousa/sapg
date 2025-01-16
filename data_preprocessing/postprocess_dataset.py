# %%
# Import necessary libraries
import json
import os

import setup_run_dir
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from data_preprocessing.preprocessing_utils import clean_sentence

dataset_name = "qqppos"  # "qqppos"  # paranmt-small
dataset_path = os.path.join("processed-data", dataset_name)
dataset_suffix = "with_amr"  # "with_amr" "with_codes" "clean_with_amr_and_codes"

# Load the dataset
processed_datasets = load_dataset(
    dataset_path,
    data_files={
        "train": f"train_{dataset_suffix}.jsonl",
        "validation": f"validation_{dataset_suffix}.jsonl",
        "test": f"test_{dataset_suffix}.jsonl",
    },
)

# %%
# Clean malformed AMR graphs
from amr_utils import convert_amr_to_graph


def try_converting_amr_to_graph(example):
    try:
        amr = convert_amr_to_graph(example["src_amr"])
        if amr is None:
            return {"parsed_amr": False}
        return {"parsed_amr": True}
    except Exception as e:
        print(f"Error converting AMR: {e}")
        return {"parsed_amr": False}


processed_datasets = processed_datasets.map(
    try_converting_amr_to_graph,
    batched=False,
    load_from_cache_file=False,
    desc="Parsing AMR to Penman graph",
)

# %%
# Error analysis
# amr_one_liner = '(a / and :op1 (a2 / and :op1 (c / case-04 :ARG1 (p / product :name (n / name :op1 "26761" :op2 "40" :op3 "0"))) :op2 (c2 / case-04 :ARG1 (p2 / product :name (n2 / name :op1 "68515 49 1 ) ) ) :op2 ( and_2 :op1 ( case-04_2 :ARG1 ( product_2 :name ( name_2 :op1 ")) :op2 (c3 / case-04 :ARG1 (p3 / product :name (n3 / name :op1 "271 091 4 ) ) ) ) :op3 ( phthalate :name ( name_4 :op1 "))))))'
# amr = convert_amr_to_graph(amr_one_liner)


# %%
# Filter out malformed entries
clean_datasets = processed_datasets.filter(lambda example: example["parsed_amr"])


# %%
def clean_source_target(examples):
    examples["source"] = clean_sentence(examples["source"])
    examples["target"] = clean_sentence(examples["target"])
    return examples


clean_datasets = clean_datasets.map(clean_source_target)


# %%
# def filter_long_sentences(example):
#     source_tokens = word_tokenize(example["source"])
#     return len(source_tokens) <= 20


# clean_datasets = clean_datasets.filter(filter_long_sentences)

# %%
# Drop parsed_amr
clean_datasets = clean_datasets.remove_columns("parsed_amr")


# %%
def save_dataset_to_jsonl(dataset, filepath):
    df = dataset.to_pandas()
    df.to_json(filepath, orient="records", lines=True)


for split in clean_datasets:
    save_dataset_to_jsonl(
        clean_datasets[split],
        f"{dataset_path}/{split}_clean_amr.jsonl",
    )

# %%
# # Add index to each example
# raw_datasets = raw_datasets.map(lambda example, idx: {"index": idx}, with_indices=True)

# # %%
# # Initialize lists to keep track of removed indices
# removed_indices_cleaning = []
# removed_indices_edge_triples = []


# # Clean malformed entries and track removed indices
# def filter_malformed(example):
#     if all(value is not None for value in example.values()):
#         return True
#     removed_indices_cleaning.append(example["index"])
#     return False


# clean_datasets = raw_datasets.filter(filter_malformed)


# # %%
# with open(f"missing_indices_{dataset_name}.json", "r") as f:
#     missing_indices = json.load(f)

#     if "index" not in processed_datasets["train"].column_names:
#         processed_datasets = processed_datasets.map(
#             lambda _, idx: {"index": idx}, with_indices=True
#         )

#     # Define a filter function to exclude missing indices
#     def filter_missing_indices(example, split_name):
#         index = example["index"]
#         return index not in missing_indices.get(split_name, [])

#     # Apply the filter to each split in the dataset
#     for split_name in processed_datasets.keys():
#         processed_datasets[split_name] = processed_datasets[split_name].filter(
#             lambda example: filter_missing_indices(example, split_name),
#             batched=False,
#             num_proc=None,
#             load_from_cache_file=False,
#             desc=f"Removing missing indices for {split_name} split",
#         )
