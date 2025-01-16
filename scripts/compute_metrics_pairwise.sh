predictions_dir="output"
model_name="t5-large" # "google/t5-v1_1-xl" # "t5-small"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
adapter_dim=256
graph_encoder="gcn"
arch_type="decoder_attention"
graph_type="amr"
metrics="my_metric"
checkpoint="$model_name-$dataset-$lr-${epochs}e-$graph_encoder-$arch_type-$adapter_dim-$graph_type"
best_epoch=1

# checkpoint="$model_name-$dataset-$lr-${epochs}e-linearized-amr"

output_file="$predictions_dir/$checkpoint/epoch_$best_epoch/pairwise_metrics.csv"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=0 python compute_metrics.py \
        --input_path $predictions_dir/$checkpoint/epoch_$best_epoch/eval_generations-amr.csv \
        --source_column source \
        --target_column target \
        --predictions_column prediction \
        --metric metrics/$metrics \
        --output_path $output_file \
        --compute_pair_wise
        "
    eval $job
fi