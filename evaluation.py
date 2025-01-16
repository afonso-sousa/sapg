"""
Evaluation script.
"""

import argparse
import json
import logging
import os
from functools import partial

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from graph_collator import GraphDataCollatorForSeq2Seq
from graph_decoder_attention_t5 import GraphTextDualEncoderT5
from graph_encoder_attention_t5 import GraphTextFusedEncoderT5
from struct_adapt import StructAdapt
from utils import (
    filter_examples_with_missing_edges,
    parse_amr_into_reduced_form,
    processing_function_for_text_and_AMR,
    processing_function_for_text_graph,
    standard_processing_function,
)

logger = get_logger(__name__)


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model for paraphrase generation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--splits_suffix",
        type=str,
        default=None,
        help="The suffix of the dataset splits.",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Whether to use append linearized semantic graph as input",
    )
    parser.add_argument(
        "--code_columns",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="dp",
        choices=["dp", "amr"],
        help="Specify the graph type (dependency parsing or amr)",
    )
    parser.add_argument(
        "--linearize",
        action="store_true",
        help="Whether to linearize the input graph",
    )
    parser.add_argument(
        "--arch_type",
        type=str,
        choices=["encoder_attention", "decoder_attention", "struct_adapt"],
        help="Specify the architecture type",
    )
    parser.add_argument(
        "--graph_encoder",
        type=str,
        default=None,
        choices=["gat", "gcn", "rgcn"],
        help="Specify the graph encoder (GAT, GCN or RGCN)",
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=256,
        help="Hidden dimension of GNN adapter.",
    )
    parser.add_argument(
        "--with_lora_weights",
        action="store_true",
        help="Whether to use load the model with LoRA weights.",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        help="The name of the base model when using LoRA weights.",
    )
    parser.add_argument(
        "--with_adapter_weights",
        action="store_true",
        help="Whether to load the model with adapter weights.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        default="eval_results.json",
        help="Where to store the final scores.",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Where to store the final predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=3,
        help="Number of beam groups.",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=None,
        help="Number of beam groups.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="The parameter for repetition penalty.",
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=None,
        help="The parameter for diversity penalty.",
    )
    args = parser.parse_args()

    # Sanity checks

    if args.dataset_name is None:
        raise ValueError("Make sure to provide a dataset name")

    if args.model_name_or_path is None:
        raise ValueError("Make sure to provide a model name")

    if args.with_graph and args.graph_encoder is None:
        raise ValueError("Set the desired graph type to use graph encoder.")

    if args.graph_encoder and not args.with_graph:
        raise ValueError("Set --with_graph to true to use the graph_encoder argument.")

    if args.with_graph and args.code_columns is not None:
        raise ValueError("Cannot use --code_columns with --with_graph.")

    if args.code_columns is not None and args.codes is None:
        raise ValueError("Specify --codes while using --code_columns.")

    if args.with_lora_weights and not args.base_model_name:
        raise ValueError(
            "Provide the base model name to load the model with LoRA weights."
        )

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger.info(f"Loading '{args.dataset_name}' dataset")
    raw_datasets = load_dataset(
        args.dataset_name,
        data_files={
            "test": f"test_{args.splits_suffix}.jsonl",
        },
    )

    if args.code_columns is not None:
        code_columns = json.loads(args.code_columns)
        codes = args.codes.split("-")

    if args.with_lora_weights:
        logger.info(f"Loading base model '{args.base_model_name}'")
        config = AutoConfig.from_pretrained(args.base_model_name)
    else:
        logger.info(f"Loading checkpoint '{args.model_name_or_path}'")
        config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    if args.with_graph:
        if args.with_adapter_weights:
            if args.arch_type == "encoder_attention":
                model = GraphTextFusedEncoderT5.from_pretrained(
                    args.model_name_or_path, config=config, ignore_mismatched_sizes=True
                )
            elif args.arch_type == "decoder_attention":
                model = GraphTextDualEncoderT5.from_pretrained(
                    args.model_name_or_path, config=config, ignore_mismatched_sizes=True
                )
            elif args.arch_type == "struct_adapt":
                model = StructAdapt.from_pretrained(
                    args.model_name_or_path, config=config, ignore_mismatched_sizes=True
                )
            else:
                raise ValueError("Unknown architecture type")

            # Load embedding weights
            load_model(
                model.shared,
                os.path.join(args.output_dir, "embedding_weights.safetensors"),
            )

            # Load adapter weights for encoder block 0
            load_model(
                model.encoder.block[0].adapter,
                os.path.join(args.output_dir, "encoder_adapter_weights.safetensors"),
            )

            # Load adapter weights for decoder block 0
            load_model(
                model.decoder.block[0].adapter,
                os.path.join(args.output_dir, "decoder_adapter_weights.safetensors"),
            )
        else:
            if args.arch_type == "encoder_attention":
                model = GraphTextFusedEncoderT5.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                )
            elif args.arch_type == "decoder_attention":
                model = GraphTextDualEncoderT5.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                )
            elif args.arch_type == "struct_adapt":
                model = StructAdapt.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                )
            else:
                raise ValueError("Unknown architecture type")
    else:
        if args.with_lora_weights:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.base_model_name,
                config=config,
            )
            model = PeftModel.from_pretrained(
                base_model,
                args.model_name_or_path,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                config=config,
            )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Temporarily set max_target_length for training.
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    glossary = None
    if args.graph_type == "amr":
        with accelerator.main_process_first():
            from glossary import AMR_LABELS_PARANMT_TRAIN, AMR_LABELS_QQP_TRAIN

            glossary_dict = {
                "paranmt": AMR_LABELS_PARANMT_TRAIN,
                "qqp": AMR_LABELS_QQP_TRAIN,
            }
            for name, glossary_var in glossary_dict.items():
                if name in args.dataset_name.lower():
                    glossary = glossary_var
                    break
            else:
                raise ValueError("Unknown graph type")

            parse_amr_with_extra_arg = partial(
                parse_amr_into_reduced_form,
                out_graph_type=(
                    "bipartite"
                    if args.graph_encoder is None
                    or args.graph_encoder in ["gat", "gcn"]
                    else "multirelational"
                ),
                rel_glossary=(glossary if args.graph_encoder == "rgcn" else None),
            )
            raw_datasets = raw_datasets.map(
                parse_amr_with_extra_arg,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Parsing AMR into reduced form",
            )

            raw_datasets = raw_datasets.filter(
                filter_examples_with_missing_edges,
                num_proc=args.preprocessing_num_workers,
                desc="Filtering examples with missing node edges",
            )

    column_names = raw_datasets["test"].column_names

    if args.with_graph:
        if args.graph_type == "dp":
            preprocess_function = processing_function_for_text_graph(
                tokenizer,
                max_source_length,
                max_target_length,
                graph_representation=(
                    "bipartite"
                    if args.graph_encoder in ["gat", "gcn"]
                    else "multirelational"
                ),
                use_linearized_graph=args.linearize,
            )
        else:
            # AMR processing function
            preprocess_function = processing_function_for_text_and_AMR(
                tokenizer,
                max_source_length,
                max_target_length,
                graph_representation=(
                    "bipartite"
                    if args.graph_encoder in ["gat", "gcn"]
                    else "multirelational"
                ),
                use_linearized_graph=args.linearize,
            )
    else:
        if not args.linearize:
            preprocess_function = standard_processing_function(
                tokenizer,
                max_source_length,
                max_target_length,
                use_linearized_graph=args.linearize,
                code_columns=code_columns if args.code_columns is not None else None,
                inference_codes=codes if args.code_columns is not None else None,
            )
        else:
            preprocess_function = processing_function_for_text_and_AMR(
                tokenizer,
                max_source_length,
                max_target_length,
                graph_representation=(
                    "bipartite"
                    if args.graph_encoder is None
                    or args.graph_encoder in ["gat", "gcn"]
                    else "multirelational"
                ),
                use_linearized_graph=args.linearize,
                only_text=True,
            )

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=not (args.with_graph or args.linearize),
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Malformed entries will have None values
    processed_datasets = processed_datasets.filter(
        lambda example: all(value is not None for value in example.values())
    )

    eval_dataset = processed_datasets["test"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # DataLoaders creation:
    label_pad_token_id = -100

    if args.with_graph:
        data_collator = GraphDataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    # Prepare everything with `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    metric = evaluate.load("metrics/my_metric")

    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}"
    )
    logger.info(
        f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )

    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": (
            args.val_max_target_length if args is not None else config.max_length
        ),
        "num_beams": args.num_beams,
    }

    if args.num_return_sequences > 1:
        multi_gen_kwargs = {
            "num_beams": args.num_return_sequences * args.beam_width,
            "num_return_sequences": args.num_return_sequences,
            "num_beam_groups": args.num_beam_groups,
            "repetition_penalty": args.repetition_penalty,
            "diversity_penalty": args.diversity_penalty,
        }
        gen_kwargs = {**gen_kwargs, **multi_gen_kwargs}

    samples_seen = 0
    sources = []
    references = []
    predictions = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            ### Predictions generation and processing ###
            if args.with_graph:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    graphs=batch["graphs"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            ### Target processing ###
            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.gather(labels).cpu().numpy()
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # Expand labels to match the size of generated_tokens
            expanded_labels = np.repeat(
                labels, len(generated_tokens) // len(labels), axis=0
            )

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                expanded_labels, skip_special_tokens=True
            )

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    decoded_labels = decoded_labels[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += len(decoded_labels)

            original_inputs = raw_datasets["test"]["source"][
                step * total_batch_size : (step + 1) * total_batch_size
            ]

            # Expand labels to match the size of generated_tokens
            expanded_inputs = np.repeat(
                original_inputs, len(generated_tokens) // len(original_inputs), axis=0
            )

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            sources.extend(expanded_inputs)

            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
                sources=expanded_inputs,
            )

    eval_metric = metric.compute()
    logger.info({"bleu": eval_metric["bleu"]})
    logger.info({"self_bleu": eval_metric["self_bleu"]})
    logger.info({"ibleu": eval_metric["ibleu"]})
    logger.info({"sbert": f"{eval_metric['sbert_mean']} +- {eval_metric['sbert_std']}"})

    if args.output_dir is not None:
        if args.scores_file:
            with open(os.path.join(args.output_dir, args.scores_file), "w") as f:
                json.dump(
                    {
                        "bleu": eval_metric["bleu"],
                        "self_bleu": eval_metric["self_bleu"],
                        "ibleu": eval_metric["ibleu"],
                        "sbert": f"{eval_metric['sbert_mean']} +- {eval_metric['sbert_std']}",
                    },
                    f,
                )

        if args.predictions_file and args.max_eval_samples is None:
            result = pd.DataFrame(
                {
                    "source": sources,
                    "target": references,
                    "prediction": predictions,
                }
            )
            result.to_csv(
                os.path.join(args.output_dir, args.predictions_file),
                index=False,
                sep="\t",
            )


if __name__ == "__main__":
    main()
