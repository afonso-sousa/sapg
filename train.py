"""
Fine-tuning a 🤗 Transformers model on paraphrase generation.
"""

import argparse
import json
import logging
import math
import os
import random
from collections import OrderedDict
from functools import partial

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

from glossary import DEPENDENCIES
from graph_collator import GraphDataCollatorForSeq2Seq
from graph_decoder_attention_t5 import GraphTextDualEncoderT5
from graph_encoder_attention_t5 import GraphTextFusedEncoderT5
from struct_adapt import StructAdapt
from utils import (
    filter_examples_with_missing_edges,
    find_double_root_nodes,
    freeze_embeds,
    freeze_original_parameters,
    freeze_params,
    freeze_params_except_adapter,
    normalize_code,
    parse_amr_into_reduced_form,
    processing_function_for_text_and_AMR,
    processing_function_for_text_graph,
    standard_processing_function,
)

logger = get_logger(__name__)


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
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
        help="Whether to use a graph as input",
    )
    parser.add_argument(
        "--code_columns",
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
        default=512,
        help="Hidden dimension of GNN adapter.",
    )
    parser.add_argument(
        "--with_lora",
        action="store_true",
        help="Whether to use LoRA to finetune.",
    )
    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="Whether to generate predictions for validation.",
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
        default=1024,
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
        "--freeze_embeddings",
        action="store_true",
        help="Whether to freeze the embedding layers' parameters.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze the encoder parameters.",
    )
    parser.add_argument(
        "--freeze_original_model",
        action="store_true",
        help="Whether to freeze every regular model parameters.",
    )
    parser.add_argument(
        "--freeze_model",
        action="store_true",
        help="Whether to freeze the model.",
    )
    parser.add_argument(
        "--use_custom_graph_similarity_loss",
        action="store_true",
        help="Whether to use custom graph similarity loss.",
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
        help="Path to pretrained model or model identifier.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_epochs_without_improvement",
        type=int,
        default=None,
        help="Maximum number of epochs without improvement before interrupting training.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="Evaluate every X epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need training/validation data.")

    if args.model_name_or_path is None:
        raise ValueError("Please provide a model name")

    if (
        args.max_epochs_without_improvement is not None
        and args.evaluation_interval != 1
    ):
        raise ValueError("--evaluation_interval must be 1 when using early stopping.")

    if args.with_graph and args.graph_encoder is None:
        raise ValueError("Set the desired graph type to use graph encoder.")

    if args.graph_encoder and not args.with_graph:
        raise ValueError("Set --with_graph to true to use the graph_encoder argument.")
    if args.with_graph and args.code_columns is not None:
        raise ValueError("Cannot use --code_columns with --with_graph")

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

    raw_datasets = load_dataset(
        args.dataset_name,
        data_files={
            "train": f"train_{args.splits_suffix}.jsonl",
            "validation": f"validation_{args.splits_suffix}.jsonl",
        },
    )

    if args.graph_type == "dp" and (args.with_graph or args.linearize):
        # remove entries with double ROOT nodes
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.filter(
                find_double_root_nodes,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Removing double ROOT nodes",
            )

    if args.code_columns is not None:
        code_columns = json.loads(args.code_columns)
        all_codes = []
        for column in code_columns:
            codes = raw_datasets["train"].unique(column)
            codes = [normalize_code(c, column) for c in codes]
            all_codes += codes
        all_codes = sorted(all_codes)
        logger.info(f"Full conditions list: {all_codes}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    glossary = None
    if args.with_graph:
        if args.graph_encoder == "rgcn":
            if args.graph_type == "dp":
                all_dependencies = OrderedDict(DEPENDENCIES).keys()
                num_relations = len(all_dependencies)
            elif args.graph_type == "amr":
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
                num_relations = len(glossary)
            config.num_relations = num_relations

        config.adapter_dim = args.adapter_dim
        config.arch_type = args.arch_type
        config.graph_encoder = args.graph_encoder

    if args.graph_type == "amr":
        with accelerator.main_process_first():
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

    special_tokens = set()
    if args.graph_encoder in ["gat", "gcn"] or (
        args.linearize and not args.graph_encoder == "rgcn"
    ):
        if args.graph_type == "dp":
            special_tokens = set(f":{dep}" for dep in DEPENDENCIES.keys())
        else:
            special_tokens.update(
                [
                    el
                    for node_tokens in raw_datasets["train"]["node_tokens"]
                    if node_tokens is not None
                    for el in node_tokens
                    if el.startswith(":") and len(el) > 1
                ]
            )
    elif args.code_columns is not None:
        special_tokens = set(all_codes)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        extra_ids=0,  # no need for sentinel tokens
        additional_special_tokens=list(special_tokens),
        use_fast=not args.use_slow_tokenizer,
    )

    if args.with_graph:
        if args.arch_type == "encoder_attention":
            model = GraphTextFusedEncoderT5.from_pretrained(
                args.model_name_or_path, config=config
            )
        elif args.arch_type == "decoder_attention":
            model = GraphTextDualEncoderT5.from_pretrained(
                args.model_name_or_path, config=config
            )
        elif args.arch_type == "struct_adapt":
            model = StructAdapt.from_pretrained(args.model_name_or_path, config=config)
        else:
            raise ValueError("Unknown architecture type")
    elif args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    if args.with_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )

        model = get_peft_model(model, lora_config)

    if args.freeze_embeddings:
        freeze_embeds(model)

    if args.freeze_encoder:
        freeze_params(model.get_encoder())

    if args.freeze_model:
        freeze_params_except_adapter(model)

    if args.freeze_original_model:
        freeze_original_parameters(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params:_}")
    logger.info(f"Total number of parameters: {total_params:_}")

    # T5-Large number of parameters: 737,668,096

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Temporarily set max_target_length for training.
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

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
            print("Using AMR preprocessing function")
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

    if args.dataset_name == "paranmt-small":
        # Some entries are malformed, we skip them.
        indices_to_keep = list(range(94640)) + list(
            range(94644, len(raw_datasets["train"]))
        )
        raw_datasets["train"] = raw_datasets["train"].select(indices_to_keep)

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=not (args.with_graph or args.linearize),
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running preprocess function on dataset",
        )

    # start = 94640  # Start of the dataset
    # end = 94643  # End of the dataset (or any specific point)
    # batch_size = 2

    # for batch_start in range(start, end, batch_size):
    #     batch_end = min(batch_start + batch_size, end)
    #     print(f"Processing batch from {batch_start} to {batch_end}")

    #     # Process the current chunk
    #     try:
    #         with accelerator.main_process_first():
    #             processed_batch = (
    #                 raw_datasets["train"]
    #                 .select(range(batch_start, batch_end))
    #                 .map(
    #                     preprocess_function,
    #                     batched=not (args.with_graph or args.linearize),
    #                     num_proc=args.preprocessing_num_workers,
    #                     remove_columns=column_names,
    #                     load_from_cache_file=not args.overwrite_cache,
    #                     desc=f"Processing batch from {batch_start} to {batch_end}",
    #                 )
    #             )
    #     except Exception as e:
    #         print(f"Error in batch from {batch_start} to {batch_end}: {e}")
    #         # Stop processing and allow you to investigate
    #         break

    # Malformed entries will have None values
    processed_datasets = processed_datasets.filter(
        lambda example: all(value is not None for value in example.values())
    )

    train_dataset = processed_datasets["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = processed_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100

    if args.with_graph:
        data_collator = GraphDataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
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

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            # breakpoint()
            # accelerator.load_state(args.resume_from_checkpoint)
            accelerator.unwrap_model(model).load_state_dict(
                torch.load(f"{args.resume_from_checkpoint}/pytorch_model.bin")
            )
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    best_score = -1
    epochs_without_improvement = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if epoch % args.evaluation_interval == 0:
            model.eval()
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": (
                    args.val_max_target_length
                    if args is not None
                    else config.max_length
                ),
                "num_beams": args.num_beams,
            }
            samples_seen = 0
            for step, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
                with torch.no_grad():
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
                        generated_tokens,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    labels = batch["labels"]

                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"],
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )
                    labels = accelerator.gather(labels).cpu().numpy()

                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    decoded_labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )

                    decoded_preds, decoded_labels = postprocess_text(
                        decoded_preds, decoded_labels
                    )

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

                    metric.add_batch(
                        predictions=decoded_preds, references=decoded_labels
                    )
            eval_metric = metric.compute()
            current_score = eval_metric["score"]

            if current_score > best_score:
                best_score = current_score
                if args.output_dir is not None:
                    logger.info(f"Saving model at epoch {epoch}")
                    output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
                    os.makedirs(output_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    if args.freeze_model or args.freeze_encoder:
                        unwrapped_model.config.architectures = [
                            unwrapped_model.__class__.__name__
                        ]
                        unwrapped_model.config.to_json_file(
                            os.path.join(output_dir, "config.json")
                        )
                        # Save embedding weights
                        save_model(
                            unwrapped_model.shared,
                            os.path.join(output_dir, "embedding_weights.safetensors"),
                        )
                        # Save adapter weights from encoder block 0
                        save_model(
                            unwrapped_model.encoder.block[0].adapter,
                            os.path.join(
                                output_dir, "encoder_adapter_weights.safetensors"
                            ),
                        )
                        # Save adapter weights from decoder block 0
                        save_model(
                            unwrapped_model.decoder.block[0].adapter,
                            os.path.join(
                                output_dir, "decoder_adapter_weights.safetensors"
                            ),
                        )
                    else:
                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                        json.dump({"eval_bleu": eval_metric["score"]}, f)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info({"bleu": current_score})

            if epochs_without_improvement >= args.max_epochs_without_improvement:
                logger.info(
                    f"No improvement for {args.max_epochs_without_improvement} epochs. Stopping training."
                )
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)


if __name__ == "__main__":
    main()
