model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
patience=6
adapter_dim=256
output_dir="output"
splits_suffix="clean_with_amr_and_codes"
graph_type="amr"
graph_representation="bipartite" # "multirelational", "bipartite"
arch_type="decoder_attention" # "encoder_attention", "decoder_attention", "struct_adapt"
output_file="$model_name-$dataset-$lr-${epochs}e-$graph_representation-$arch_type-$adapter_dim-$graph_type"
training_setup="gnn_plus_decoder" # "full" "gnn_only" "gnn_plus_decoder"

if [ "$training_setup" = "gnn_only" ]; then
    output_file="$output_file-gnn_only"
    freeze_option="--freeze_model"
elif [ "$training_setup" = "gnn_plus_decoder" ]; then
    output_file="$output_file-gnn_plus_decoder"
    freeze_option="--freeze_original_model"
else
    freeze_option=""
fi

# accelerate launch 
# CUDA_VISIBLE_DEVICES=0 python 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path $model_name \
    --with_graph \
    --linearize \
    --graph_type $graph_type \
    --arch_type $arch_type \
    --graph_representation $graph_representation \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $output_dir/$output_file \
    --num_train_epochs $epochs \
    --max_eval_samples 10 \
    --evaluation_interval 1 \
    --adapter_dim $adapter_dim \
    --num_warmup_steps 100 \
    --learning_rate $lr \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --max_source_length 256 \
    --max_target_length 64 \
    $freeze_option \
    --max_epochs_without_improvement $patience \
    --gradient_accumulation_steps 2
    # --resume_from_checkpoint $output_dir/$output_file/epoch_5
