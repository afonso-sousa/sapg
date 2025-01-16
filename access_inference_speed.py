# %%
import argparse
import copy
import statistics
import time

import amrlib
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from graph_collator import GraphDataCollatorForSeq2Seq
from graph_decoder_attention_t5 import GraphTextDualEncoderT5
from utils import (
    create_bipartite_graph_from_amr_inputs,
    parse_amr_into_reduced_form,
    processing_function_for_text_graph,
    standard_processing_function,
)

dataset = "qqppos"  # "paranmt-small"  # "qqppos"  # "paws"
with_graph = False
linearize = True
model_name = "output/t5-large-qqppos-1e-4-6e-gcn-decoder_attention-256-amr/epoch_1"

args1 = argparse.Namespace(
    model_name_or_path=model_name,
    with_graph=True,
    linearize_amr=True,
    use_slow_tokenizer=False,
    arch_type="decoder_attention",  # "encoder_attention",
    graph_encoder="gcn",
    max_source_length=128,
    max_target_length=128,
    num_beams=4,
)
config1 = AutoConfig.from_pretrained(args1.model_name_or_path)
tokenizer1 = AutoTokenizer.from_pretrained(
    args1.model_name_or_path, use_fast=not args1.use_slow_tokenizer
)
model1 = GraphTextDualEncoderT5.from_pretrained(
    args1.model_name_or_path, config=config1
)
model1.eval()


# %%
# Access time to extract AMR from a sentence


src1 = "What are the prospect of mechanical engineers in future?"

# Load pretrained StoG model and generate AMR
stog = amrlib.load_stog_model(
    model_dir="data_preprocessing/model_parse_xfm_bart_large-v0_1_0",
    device="cuda:0",
    batch_size=4,
)

num_iterations = 100
total_time = 0
for i in tqdm(range(num_iterations)):
    start_time = time.time()
    amr_str = stog.parse_sents([src1])[0]
    amr_str = amr_str.split("\n", 1)[1]
    end_time = time.time()
    processing_time = end_time - start_time

    total_time += processing_time

avg_time = total_time / num_iterations
print(
    f"Average processing time over {num_iterations} iterations: {avg_time:.4f} seconds"
)

# %%
# Processing AMR into reduced form
accelerator = Accelerator()
src_amr1 = "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"
num_iterations = 100
total_time = 0
for i in tqdm(range(num_iterations)):
    start_time = time.time()
    example = parse_amr_into_reduced_form(
        {"src_amr": src_amr1},
        out_graph_type="bipartite",
        rel_glossary=None,
    )
    model_inputs = tokenizer1(
        example["node_tokens"],
        max_length=args1.max_source_length,
        padding=False,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    graph_inputs = create_bipartite_graph_from_amr_inputs(
        example,
        args1.max_source_length,
        tokenizer1,
    )
    graphs = [
        Data(
            x=torch.tensor(graph_inputs["x"], dtype=torch.long),
            edge_index=torch.tensor(graph_inputs["edge_index"], dtype=torch.long)
            .t()
            .contiguous(),
        )
    ]
    model_inputs["graphs"] = Batch.from_data_list(graphs).to(accelerator.device)

    # Ensure all tensors are moved to the correct device
    model_inputs = {
        key: value.to(accelerator.device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }

    # Prepare model and inputs with accelerator
    model1, model_inputs = accelerator.prepare(model1, model_inputs)

    end_time = time.time()
    processing_time = end_time - start_time

    total_time += processing_time

avg_time = total_time / num_iterations
print(
    f"Average processing time over {num_iterations} iterations: {avg_time:.4f} seconds"
)

# %%
# Generate the predictions
gen_kwargs = {
    "max_length": (
        args1.max_target_length if args1 is not None else config1.max_length
    ),
    "num_beams": args1.num_beams,
}

num_iterations = 100
total_time = 0
for i in tqdm(range(num_iterations)):
    start_time = time.time()
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model1).generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            graphs=copy.deepcopy(model_inputs["graphs"]),
            **gen_kwargs,
        )

    # Post-process the generations
    generated_tokens = accelerator.pad_across_processes(
        generated_tokens, dim=1, pad_index=tokenizer1.pad_token_id
    )
    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    decoded_preds = tokenizer1.batch_decode(generated_tokens, skip_special_tokens=True)

    end_time = time.time()
    processing_time = end_time - start_time

    total_time += processing_time

avg_time = total_time / num_iterations
print(
    f"Average processing time over {num_iterations} iterations: {avg_time:.4f} seconds"
)

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model2 = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

model_inputs = tokenizer1(
    src1,
    max_length=args1.max_source_length,
    padding=False,
    truncation=True,
    return_tensors="pt",
)

gen_kwargs = {
    "max_length": (
        args1.max_target_length if args1 is not None else config1.max_length
    ),
    "num_beams": args1.num_beams,
}

num_iterations = 100
total_time = 0
for i in tqdm(range(num_iterations)):
    start_time = time.time()
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model2).generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            **gen_kwargs,
        )

    # Post-process the generations
    generated_tokens = accelerator.pad_across_processes(
        generated_tokens, dim=1, pad_index=tokenizer1.pad_token_id
    )
    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    decoded_preds = tokenizer1.batch_decode(generated_tokens, skip_special_tokens=True)

    end_time = time.time()
    processing_time = end_time - start_time

    total_time += processing_time

avg_time = total_time / num_iterations
print(
    f"Average processing time over {num_iterations} iterations: {avg_time:.4f} seconds"
)

# %%
