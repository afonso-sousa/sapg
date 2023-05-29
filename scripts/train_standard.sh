model_name="t5-small"
dataset="data/paws"
output_file="output/$model_name-$dataset-standard"

python train.py \
    --model_name_or_path $model_name \
    --dataset_name $dataset \
    --output_dir $output_file