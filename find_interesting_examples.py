# %%
import pandas as pd

top_x = 5
dataset_name = "paranmt-small"  # "qqppos" # "paws" # "paranmt-small"
file_paths = [
    f"output/t5-small-{dataset_name}-1e-4-100e-concat-bipartite-first-512-no_freeze/pairwise_metrics.csv",
    f"output/t5-small-{dataset_name}-1e-4-100e-linearized/pairwise_metrics.csv",
    f"output/t5-small-{dataset_name}-standard/pairwise_metrics.csv",
]

dfs = [pd.read_csv(file_path, sep="\t")[:-1] for file_path in file_paths]

# %%
diff_rows = []
for idx in dfs[0].index:
    bleu_values = [df.loc[idx, "bleu"] for df in dfs]
    if all(value != 100 for value in bleu_values):
        self_bleu_values = [df.loc[idx, "self_bleu"] for df in dfs]
        if all(value != 100 for value in self_bleu_values):
            ibleu_values = [df.loc[idx, "ibleu"] for df in dfs]
            diff = max(ibleu_values) - min(ibleu_values)
            diff_rows.append((idx, diff))

top_diff_rows = sorted(diff_rows, key=lambda x: x[1], reverse=True)[:top_x]

# %%
# Accessing "prediction" headers for the top-X rows
top_predictions = [
    [dfs[x].loc[row[0], "prediction"] for x in range(len(dfs))] for row in top_diff_rows
]

top_src_and_tgt = [dfs[0].loc[row[0], ["source", "target"]] for row in top_diff_rows]
top_src_and_tgt_df = pd.DataFrame(top_src_and_tgt, columns=["source", "target"])

df_top_predictions = pd.DataFrame(
    top_predictions, columns=["adapter", "linearized", "standard"]
)

df_top_predictions = pd.concat(
    [top_src_and_tgt_df.reset_index(drop=True), df_top_predictions], axis=1
)
print(df_top_predictions)

# %%
