model_name="t5-large"
data_dir="processed-data"
dataset="paranmt-small" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
graph_type="amr" # "dp"
checkpoint="$model_name-$dataset-$lr-${epochs}e-linearized-$graph_type"
scores_file="multi_eval_scores.csv"
predictions_file="eval_multi_generations.csv"
splits_suffix="clean_with_amr_and_codes" # "with_dep_tree"
best_epoch=1

if [ -n "$best_epoch" ]; then
    checkpoint=$checkpoint/epoch_$best_epoch
fi

model_dir=output/${checkpoint}

# accelerate launch 
CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --linearize \
    --graph_type $graph_type \
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
