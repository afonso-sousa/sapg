model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # "qqppos" # "paws" # "paranmt-small"
epochs=6
lr="1e-4"
adapter_dim=256
graph_representation="bipartite" # "multirelational", "bipartite"
arch_type="decoder_attention" # "encoder_attention", "decoder_attention", "struct_adapt"
graph_type="amr"
checkpoint="$model_name-$dataset-$lr-${epochs}e-$graph_representation-$arch_type-$adapter_dim-$graph_type"
scores_file="eval_scores-$graph_type.csv"
predictions_file="eval_generations-$graph_type.csv"
splits_suffix="clean_with_amr_and_codes"
training_setup="gnn_plus_decoder" # "full" "gnn_only" "gnn_plus_decoder"
best_epoch=4

if [ "$training_setup" = "gnn_only" ]; then
    checkpoint="$checkpoint-gnn_only"
elif [ "$training_setup" = "gnn_plus_decoder" ]; then
    checkpoint="$checkpoint-gnn_plus_decoder"
fi

if [ -n "$best_epoch" ]; then
    checkpoint=$checkpoint/epoch_$best_epoch
fi

model_dir=output/${checkpoint}

# accelerate launch 
CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --with_graph \
    --linearize \
    --graph_type $graph_type \
    --arch_type $arch_type \
    --graph_representation $graph_representation \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file \
    --per_device_eval_batch_size 8
    #--with_adapter_weights \
    #--base_model $model_name
