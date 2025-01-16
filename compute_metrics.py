import argparse
import os

import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset


# Parsing input arguments
def parse_args():
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    parser = argparse.ArgumentParser(
        description="Compute metrics from a file of generated paraphrases"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="eval_generations.csv",
        help="Input file",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        help="the source column",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        help="the target column",
    )
    parser.add_argument(
        "--predictions_column",
        type=str,
        help="the prediction column",
    )
    parser.add_argument(
        "--compute_pair_wise",
        action="store_true",
        help="whether to compute metrics pair-wise",
    )
    parser.add_argument(
        "--lower_case",
        action="store_true",
        help="Whether to convert source and targets to lowercase.",
    )
    parser.add_argument(
        "--metric_name_or_path",
        type=str,
        help="metric you wish to apply.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pairs_evals.csv",
        help="File to output the results.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print(f"Output file: {args.output_path}")
    os.makedirs(os.path.abspath(os.path.dirname(args.output_path)), exist_ok=True)

    dataset = load_dataset(
        "csv",
        data_files=args.input_path,
        delimiter="\t",
    )
    dataset = dataset["train"]

    dataset = dataset.filter(lambda x: x["prediction"] is not None)

    sources = dataset[args.source_column]
    references = dataset[args.target_column]  # same as sources for reconstruction
    predictions = dataset[args.predictions_column]

    if args.lower_case:
        sources = list(map(str.lower, sources))
        references = list(map(str.lower, references))

    my_metric = evaluate.load(args.metric_name_or_path, experiment_id=os.getpid())

    print("Computing metric...")
    result = my_metric.compute(
        sources=sources,
        predictions=predictions,
        references=references,
        compute_pair_wise=args.compute_pair_wise,
    )

    if not args.compute_pair_wise:
        model_name = args.input_path.split("/")[-2]
        print(model_name)
        print(result)
        return

    result = {
        name: list(np.around(np.array(scores) * 100, 3))
        for name, scores in result.items()
    }

    result["source"] = sources
    result["target"] = references
    result["prediction"] = predictions

    df = pd.DataFrame(result)
    df.loc["mean"] = df.mean(numeric_only=True).round(3)
    df.to_csv(args.output_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
