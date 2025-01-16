model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
patience=6
graph_type="amr" # "dp"
output_dir="output"
splits_suffix="clean_with_amr_and_codes" # "with_dep_tree"
output_file="$model_name-$dataset-$lr-${epochs}e-linearized-$graph_type"

# accelerate launch --main_process_port 29502 
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path $model_name \
    --linearize \
    --graph_type $graph_type \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $output_dir/$output_file \
    --num_train_epochs $epochs \
    --max_eval_samples 10 \
    --evaluation_interval 1 \
    --num_warmup_steps 100 \
    --learning_rate $lr \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_epochs_without_improvement $patience
