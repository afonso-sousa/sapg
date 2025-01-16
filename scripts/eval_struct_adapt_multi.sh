model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
adapter_dim=512
graph_representation="bipartite" # "multirelational", "bipartite"
arch_type="struct_adapt" # "encoder_attention", "decoder_attention", "struct_adapt"
graph_type="amr"
checkpoint="$model_name-$dataset-$lr-${epochs}e-$graph_representation-$arch_type-$adapter_dim-$graph_type"
scores_file="multi_eval_scores-$graph_type.csv"
predictions_file="eval_multi_generations-$graph_type.csv"
splits_suffix="clean_with_amr_and_codes"
full_finetune=true
best_epoch=3

if ! $full_finetune; then
    checkpoint="$checkpoint-freeze"
fi

if [ -n "$best_epoch" ]; then
    checkpoint=$checkpoint/epoch_$best_epoch
fi

model_dir=output/${checkpoint}

# accelerate launch 
CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --linearize \
    --with_graph \
    --graph_type $graph_type \
    --arch_type $arch_type \
    --graph_representation $graph_representation \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file \
    --num_return_sequences 4 \
    --beam_width 2 \
    --num_beam_groups 2 \
    --repetition_penalty 1.2 \
    --diversity_penalty 0.3 \
    --per_device_eval_batch_size 8