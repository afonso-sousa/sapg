# SAPG: Semantically-Aware Paraphrase Generation with AMR Graphs


This repo contains the code for the paper SAPG: Semantically-Aware Paraphrase Generation with AMR Graphs, by Afonso Sousa & Henrique Lopes Cardoso (accepted at ICAART 2025).

Automatically generating paraphrases is crucial for various natural language processing tasks. Current approaches primarily try to control the surface form of generated paraphrases by resorting to syntactic graph structures. However, paraphrase generation is rooted in semantics, but there are almost no works trying to leverage semantic structures as inductive biases for the task of generating paraphrases. We propose SAPG, a semantically-aware paraphrase generation model, which encodes Abstract Meaning Representation (AMR) graphs into a pretrained language model using a graph neural network-based encoder. We demonstrate that SAPG enables the generation of more diverse paraphrases by transforming the input AMR graphs, allowing for control over the output generations' surface forms rooted in semantics. This approach ensures that the semantic meaning is preserved, offering flexibility in paraphrase generation without sacrificing fluency or coherence. Our extensive evaluation on two widely-used paraphrase generation datasets confirms the effectiveness of this method.

## Installation
First, to create a fresh conda environment with all the used dependencies run:
```
conda env create -f environment.yml
```

## Preprocess data
SAPG requires AMR data. To extract it you may run:
```
sh ./scripts/extract_amr.sh
```

## Train and test models
To train/test SAPG or any other model refered to in the paper you can run the corresponding script. For example:
```
sh ./scripts/train_graph_amr.sh
```

```
sh ./scripts/test_graph_amr.sh
```
