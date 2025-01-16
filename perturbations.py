# %%
import argparse

import penman
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch_geometric.data import Batch, Data
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from graph_decoder_attention_t5 import GraphTextDualEncoderT5
from utils import create_bipartite_graph_from_amr_inputs, parse_amr_into_reduced_form

logger = get_logger(__name__)

# %%
accelerator = Accelerator()

# qqp model
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

# qqp linearized graph model
model_name = "output/t5-large-qqppos-1e-4-6e-linearized-amr/epoch_5"

args2 = argparse.Namespace(
    model_name_or_path=model_name,
    with_graph=False,
    linearize_amr=True,
    use_slow_tokenizer=False,
    max_source_length=128,
    max_target_length=128,
    num_beams=4,
)
config2 = AutoConfig.from_pretrained(args2.model_name_or_path)
tokenizer2 = AutoTokenizer.from_pretrained(
    args2.model_name_or_path, use_fast=not args2.use_slow_tokenizer
)
model2 = AutoModelForSeq2SeqLM.from_pretrained(
    args2.model_name_or_path,
    config=config2,
)

total_params = sum(p.numel() for p in model1.parameters())
print(f"Graph Model # parameters: {total_params:_}")
total_params = sum(p.numel() for p in model2.parameters())
print(f"Linearized Graph Model # parameters: {total_params:_}")

# %%
# QQP Data
src1 = "What are the prospect of mechanical engineers in future?"
tgt1 = "What would be the mechanical engineering opportunities in future?"
src_amr1 = "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"

# %%
# Focus Change Test
pgraph = penman.decode(src_amr1)

candidate_tops = pgraph.variables()
new_graphs = [penman.encode(pgraph, top=t) for t in candidate_tops]

# %%
# Branch Editing Test
new_graphs = [
    "(a / and\n      :op1 (d / dream-01\n            :ARG0 (ii / i)\n            :time (y / wednesday))\n      :op2 (c / concern-02\n            :ARG0 d\n            :ARG1 (y2 / you)))"
]

# %%
# Branch Editing Test
new_graphs = [
    "(a / and\n     :op1 (d / dream-01\n        :ARG0 (ii / i)\n        :time (t / day\n        :quant 2\n      :mod (a / ago)))\n      :op2 (c / concern-02\n  :ARG0 d\n       :ARG1 (y2 / you)))"
]

# %%
# QQP Branch Addition Test
# "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"
new_graphs = [
    "(p / prospect-02\n   :ARG0 (p2 / person\n             :ARG0-of (e / engineer-01\n                         :ARG1 (m / mechanics\n   :mod (v / vehicle)))))))\n   :ARG1 (a / amr-unknown)\n   :time (f / future))"
]

# %%
# QQP Branch Editing Test
new_graphs = [
    "(p / prospect-02\n   :ARG0 (p2 / person\n             :ARG0-of (e / engineer-01\n                         :ARG1 (m / electronical)))\n   :ARG1 (a / amr-unknown)\n   :time (f / wednesday))"
]

# %%
# QQP Branch Removal Test (amr-unknown)
new_graphs = [
    "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :time (f / future))"
]

# %%
# QQP Branch Removal Test
new_graphs = ["(p / prospect-02\n      :ARG0 (p2 / person)\n      :time (f / future))"]

# %%
# Feeding another AMR graph
new_graphs = [
    "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"
]

# %%
from glossary import AMR_LABELS_PARANMT_TRAIN, AMR_LABELS_QQP_TRAIN

glossary = AMR_LABELS_QQP_TRAIN


print(f"Source sentence: '{src1}'")
print(f"Target sentence: '{tgt1}'")

for amr_str_g in new_graphs:
    # Parse the AMR into a reduced form
    example = parse_amr_into_reduced_form(
        {"src_amr": amr_str_g},
        out_graph_type="bipartite",
        rel_glossary=None,
    )
    model_inputs1 = tokenizer1(
        example["node_tokens"],
        # parse_amr_into_reduced_form({"src_amr": src_amr1})["node_tokens"],
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
    # Prepare the graph data
    graphs = [
        Data(
            x=torch.tensor(graph_inputs["x"], dtype=torch.long),
            edge_index=torch.tensor(graph_inputs["edge_index"], dtype=torch.long)
            .t()
            .contiguous(),
        )
    ]
    model_inputs1["graphs"] = Batch.from_data_list(graphs).to(accelerator.device)

    model_inputs2 = tokenizer2(
        example["node_tokens"],
        max_length=args2.max_source_length,
        padding=False,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    # Ensure all tensors are moved to the correct device
    model_inputs1 = {
        key: value.to(accelerator.device) if torch.is_tensor(value) else value
        for key, value in model_inputs1.items()
    }
    model_inputs2 = {
        key: value.to(accelerator.device) if torch.is_tensor(value) else value
        for key, value in model_inputs2.items()
    }

    # Prepare model and inputs with accelerator
    model1, model_inputs1 = accelerator.prepare(model1, model_inputs1)
    model2, model_inputs2 = accelerator.prepare(model2, model_inputs2)

    # Generate the predictions
    gen_kwargs = {
        "max_length": (
            args1.max_target_length if args1 is not None else config1.max_length
        ),
        "num_beams": args1.num_beams,
    }

    with torch.no_grad():
        generated_tokens1 = accelerator.unwrap_model(model1).generate(
            model_inputs1["input_ids"],
            attention_mask=model_inputs1["attention_mask"],
            graphs=model_inputs1["graphs"],
            **gen_kwargs,
        )
        generated_tokens2 = accelerator.unwrap_model(model2).generate(
            model_inputs2["input_ids"],
            attention_mask=model_inputs2["attention_mask"],
            **gen_kwargs,
        )

    print(f"AMR graph: '{amr_str_g}'")

    # Post-process the generations
    generated_tokens = accelerator.pad_across_processes(
        generated_tokens1, dim=1, pad_index=tokenizer1.pad_token_id
    )
    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    decoded_preds = tokenizer1.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Graph Model Generated output:\n\t'{decoded_preds[0]}'")

    generated_tokens = accelerator.pad_across_processes(
        generated_tokens2, dim=1, pad_index=tokenizer2.pad_token_id
    )
    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    decoded_preds = tokenizer2.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Linearized Graph Model Generated output:\n\t'{decoded_preds[0]}'")


# %%
