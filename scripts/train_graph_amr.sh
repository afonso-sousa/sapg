model_name="t5-large"
data_dir="processed-data"
dataset="paranmt-small" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
patience=6
adapter_dim=256
output_dir="output"
splits_suffix="clean_with_amr_and_codes"
graph_type="amr"
graph_encoder="gcn" # "gat", "gcn", "rgcn"
arch_type="decoder_attention" # "encoder_attention", "decoder_attention", "struct_adapt"
output_file="$model_name-$dataset-$lr-${epochs}e-$graph_encoder-$arch_type-$adapter_dim-$graph_type-no-lin"

# accelerate launch 
# CUDA_VISIBLE_DEVICES=0 python 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path $model_name \
    --with_graph \
    --graph_type $graph_type \
    --arch_type $arch_type \
    --graph_encoder $graph_encoder \
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
    --per_device_eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_epochs_without_improvement $patience
    # --resume_from_checkpoint $output_dir/$output_file/epoch_0
