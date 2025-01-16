model_name="t5-large" # "google/t5-v1_1-xl" # "t5-small"
data_dir="processed-data"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
checkpoint="$(echo "$model_name" | sed 's/[-\/]/_/g')-$dataset-codes"
scores_file="multi_eval_scores.csv"
predictions_file="eval_multi_generations.csv"
splits_suffix="with_codes"
codes="50-30"
best_epoch=1

if [ -n "$best_epoch" ]; then
    checkpoint=$checkpoint/epoch_$best_epoch
fi

model_dir=output/${checkpoint}

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --model_name_or_path $model_dir \
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
    --per_device_eval_batch_size 8 \
    --code_columns '["semantic_sim", "lexical_div"]' \
    --codes $codes
    