raw_datasets_dir="raw-data"
processed_datasets_dir="processed-data"
dataset="paranmt-small" # "paws" "qqppos" "paranmt-small"

# CUDA_VISIBLE_DEVICES=1
for split in "train" # "validation" "test"
do
    output_file=$processed_datasets_dir/$dataset/${split}_with_amr.jsonl
    command="python data_processing/extract_amr.py \
            --dataset_name $raw_datasets_dir/$dataset \
            --split $split \
            --output_file $output_file \
            --drop_exemplars"
    
    if [ ! -f "$output_file" ]; then
        eval $command
    else
        echo "The file '$output_file' already exists"
    fi
done