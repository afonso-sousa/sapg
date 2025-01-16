import argparse
import os

import evaluate
import setup_run_dir
from datasets import load_dataset

from data_preprocessing.preprocessing_utils import clean_sentence


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract semantic and lexical codes from sentences."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="The split to process.",
    )
    parser.add_argument(
        "--input_name_suffix",
        type=str,
        default=None,
        help="The split file name suffix.",
    )
    parser.add_argument(
        "--drop_exemplars",
        action="store_true",
        help="Whether to drop exemplars if exist",
    )
    parser.add_argument(
        "--clean_sentences",
        action="store_true",
        help="Whether to clean sentences",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="processed_data.jsonl",
        help="Where to store the processed data.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need to specify the data source.")

    return args


def main():
    args = parse_args()

    os.makedirs(os.path.abspath(os.path.dirname(args.output_file)), exist_ok=True)

    print(f"{args.dataset_name}")
    print(f"Output file: {args.output_file}")

    dataset = load_dataset(
        args.dataset_name,
        data_files={args.split: f"{args.split}{args.input_name_suffix}"},
    )
    dataset = dataset[args.split]

    # dataset = dataset.select(range(5))

    if "Unnamed: 0" in dataset.column_names:
        dataset = dataset.remove_columns(["Unnamed: 0"])
    if args.drop_exemplars and "exemplar" in dataset.column_names:
        dataset = dataset.remove_columns(["exemplar"])

    if args.clean_sentences:
        dataset = dataset.map(
            lambda example: {
                key: clean_sentence(value) for key, value in example.items()
            }
        )

    sources = dataset["source"]
    targets = dataset["target"]

    metric = evaluate.load("metrics/qcpg_metric")

    print("Computing metric...")
    scores = metric.compute(predictions=targets, references=sources)

    dataset = dataset.add_column(name="semantic_sim", column=scores["semantic_sim"])
    dataset = dataset.add_column(name="lexical_div", column=scores["lexical_div"])

    dataset.to_json(args.output_file, force_ascii=False)


if __name__ == "__main__":
    main()
