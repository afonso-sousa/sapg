input_datasets_dir="processed-data"
output_datasets_dir="processed-data"
dataset="paranmt-small" # "qqppos" # "paranmt-small"
output_name="clean_with_amr_and_codes"
input_name_suffix="_clean_amr.jsonl" # ".csv.gz"

for split in "train" # "validation" "test"
do
    output_file=$output_datasets_dir/$dataset/${split}_${output_name}.jsonl
    command="CUDA_VISIBLE_DEVICES=1 python data_preprocessing/extract_codes.py \
            --dataset_name $input_datasets_dir/$dataset \
            --split $split \
            --input_name_suffix $input_name_suffix \
            --output_file $output_file \
            --drop_exemplars"
    
    if [ ! -f "$output_file" ]; then
        eval $command
    else
        echo "The file '$output_file' already exists"
    fi
done