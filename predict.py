import argparse

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch_geometric.data import Batch, Data
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from graph_t5_old import GraphT5ForConditionalGeneration

logger = get_logger(__name__)

# model_name = "output/t5-small-qqppos-graph"
model_name = "output/t5-small-paws-graph"

args = argparse.Namespace(
    model_name_or_path=model_name,
    with_graph=True,
    use_slow_tokenizer=False,
    max_source_length=128,
    max_target_length=128,
    num_beams=5,
)

accelerator = Accelerator()

if args.model_name_or_path:
    config = AutoConfig.from_pretrained(args.model_name_or_path)
else:
    raise ValueError("Make sure to provide a model name")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, use_fast=not args.use_slow_tokenizer
)

if args.with_graph:
    model = GraphT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config
    )
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

test_sentence = "Pat loves Chris."
exemplar = "Chris is loved by Pat."
# test_sentence = "Was YouTube sold to Google?"
# exemplar = "Is Google buying YouTube?"

model_inputs = tokenizer(
    test_sentence,
    max_length=args.max_source_length,
    padding=False,
    truncation=True,
    return_tensors="pt",
)

if args.with_graph:
    import spacy

    nlp = spacy.load("en_core_web_lg")

    def get_dependency_structure(sentence):
        sent = nlp(sentence)
        dp = []
        for token in sent:
            dep_parse = {}
            dep_parse["token"] = token.text
            dep_parse["pos"] = token.tag_
            dep_parse["head"] = token.head.i if token.head.i != token.i else -1
            dep_parse["dep"] = token.dep_
            dp.append(dep_parse)
        return dp

    def get_dp_for_text(text):
        input_sents = nlp(text).sents
        dep_trees = []
        for sentence in input_sents:
            dep_parse = get_dependency_structure(sentence.text)
            dep_trees.append(dep_parse)

        return dep_trees

    trees = get_dp_for_text(exemplar)

    from collections import OrderedDict
    from itertools import chain

    from glossary import DEPENDENCIES, POS

    all_dependencies = list(OrderedDict(DEPENDENCIES).keys())
    all_pos = list(OrderedDict(POS).keys())
    x = []
    edge_index = []
    edge_type = []
    for i, node in enumerate(chain.from_iterable(trees)):
        x.append(all_pos.index(node["pos"]))
        if node["head"] == -1:
            continue
        edge_index.append(
            [
                i,
                node["head"],
            ]
        )
        edge_type.append(all_dependencies.index(node["dep"].upper()))

    graphs = [
        Data(
            x=torch.tensor(x, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(edge_type, dtype=torch.long),
        )
    ]

    model_inputs["graphs"] = Batch.from_data_list(graphs)

model, model_inputs = accelerator.prepare(model, model_inputs)

gen_kwargs = {
    "max_length": args.max_target_length if args is not None else config.max_length,
    "num_beams": args.num_beams,
}

with torch.no_grad():
    generated_tokens = accelerator.unwrap_model(model).generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        graphs=model_inputs["graphs"],
        **gen_kwargs,
    )

generated_tokens = accelerator.pad_across_processes(
    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
)
generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(f"Test sentence: '{test_sentence}'")
print(f"Exemplar: '{exemplar}'")
print(f"Generated output: '{decoded_preds[0]}'")  # single entry in batch
