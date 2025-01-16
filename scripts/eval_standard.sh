model_name="google/t5-v1_1-xl" # "google/t5-v1_1-xl" # "t5-small"
data_dir="processed-data"
dataset="paranmt-small" # "qqppos" # "paws" # "paranmt-small"
checkpoint="$(echo "$model_name" | sed 's/[-\/]/_/g')-$dataset-standard"
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
splits_suffix="clean_with_amr_and_codes"
best_epoch=0

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
    --per_device_eval_batch_size 2 \
    --num_return_sequences 4 \
    --beam_width 2 \
    --num_beam_groups 2 \
    --repetition_penalty 1.2 \
    --diversity_penalty 0.3 \
    --base_model_name $model_name \
    --with_lora_weights