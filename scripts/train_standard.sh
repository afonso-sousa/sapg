model_name="google/t5-v1_1-xl" # "google/t5-v1_1-xl" # "t5-small"
data_dir="processed-data"
dataset="paranmt-small" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
patience=6
output_dir="output"
output_file="$(echo "$model_name" | sed 's/[-\/]/_/g')-$dataset-standard"
splits_suffix="clean_with_amr_and_codes"

# CUDA_VISIBLE_DEVICES=0 python
# accelerate launch 
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path $model_name \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $output_dir/$output_file \
    --num_train_epochs $epochs \
    --max_eval_samples 10 \
    --evaluation_interval 1 \
    --num_warmup_steps 100 \
    --learning_rate $lr \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_epochs_without_improvement $patience \
    --with_lora
