import argparse
import os

import amrlib
import setup_run_dir
import spacy
from datasets import load_dataset
from preprocessing_utils import clean_sentence

nlp = spacy.load("en_core_web_sm")


def get_amr_for_sentences(sentences):
    # Load pretrained StoG model and generate AMR
    stog = amrlib.load_stog_model(
        model_dir="model_parse_xfm_bart_large-v0_1_0", device="cuda:0", batch_size=4
    )
    if isinstance(sentences, str):
        sentences = [sentences]
    gen_amr_strs = stog.parse_sents(sentences)
    amr_strings = []
    for amr_str in gen_amr_strs:
        gen_amr_str = amr_str.split("\n", 1)[1]
        amr_strings.append(gen_amr_str)

    return amr_strings


def add_amr_strings_to_entries(sample, key="source"):
    amr_strings = get_amr_for_sentences(sample[key])
    if key == "source":
        prefix = "src"
    else:
        prefix = "tgt"

    if isinstance(sample[key], list):
        # batched
        amr_strings_list = []
        cleaned_text = []
        for i in range(len(sample[key])):
            sentences = nlp(sample[key][i]).sents
            clean_sentences = []
            for sentence in sentences:
                clean_sent = clean_sentence(sentence.text)
                clean_sentences.append(clean_sent)
            cleaned_entry = " ".join(clean_sentences)
            cleaned_text.append(cleaned_entry)
            amr_strings_list.append(amr_strings[i])

        sample[f"{prefix}_amr"] = amr_strings_list

    else:
        sentences = nlp(sample[key]).sents
        clean_sentences = []
        for sentence in sentences:
            clean_sent = clean_sentence(sentence.text)
            clean_sentences.append(clean_sent)
        cleaned_text = " ".join(clean_sentences)
        sample[f"{prefix}_amr"] = amr_strings[0]

    sample[key] = cleaned_text

    return sample


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AMR graphs from sentences.")
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
        "--drop_exemplars",
        action="store_true",
        help="Whether to drop exemplars if exist",
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
        args.dataset_name, data_files={args.split: f"{args.split}.csv.gz"}
    )
    dataset = dataset[args.split]

    # dataset = dataset.select(range(2))

    # handle conditional columns
    if "Unnamed: 0" in dataset.column_names:
        dataset = dataset.remove_columns(["Unnamed: 0"])
    if args.drop_exemplars and "exemplar" in dataset.column_names:
        dataset = dataset.remove_columns(["exemplar"])

    processed_dataset = dataset.map(
        add_amr_strings_to_entries,
        batched=True,
        fn_kwargs={"key": "source"},
        num_proc=1,
    )
    processed_dataset = processed_dataset.map(
        add_amr_strings_to_entries,
        batched=True,
        fn_kwargs={"key": "target"},
        num_proc=1,
    )
    processed_dataset.to_json(args.output_file, force_ascii=False)


if __name__ == "__main__":
    main()
