predictions_dir="output"
model_name="t5-large"
dataset="paranmt-small" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
adapter_dim=256
graph_encoder="gcn"
arch_type="decoder_attention"
graph_type="amr"
metrics="my_metric"
checkpoint="$model_name-$dataset-$lr-${epochs}e-$graph_encoder-$arch_type-$adapter_dim-$graph_type"
best_epoch=0
output_file="$predictions_dir/$checkpoint/epoch_$best_epoch/gold_metrics.csv"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=1 python compute_metrics.py \
            --input_path $predictions_dir/$checkpoint/epoch_$best_epoch/eval_generations-amr.csv \
            --source_column source \
            --target_column target \
            --predictions_column source \
            --metric metrics/$metrics \
            --output_path $output_file \
        "
    eval $job
fi            
