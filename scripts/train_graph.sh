model_name="t5-small"
data_dir="data"
dataset="paws"
output_dir="output"
output_file="$model_name-$dataset-graph"

python train.py \
    --model_name_or_path $model_name \
    --with_graph \
    --dataset_name $data_dir/$dataset \
    --output_dir $output_dir/$output_file