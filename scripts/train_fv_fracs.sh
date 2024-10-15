frac_values=(0.0 1.0)
dataset_name="scifact"
model="gpt-3.5-turbo"
cot="False"
model_name="meta-llama/Meta-Llama-3.1-8B --peft lora"


source proj_params.sh
declare -A max_len_dict=( ["factify"]=800 ["wanli"]=150 ["scifact"]=512 ["fever"]=100 )
declare -A size_dict=( ["factify"]=1000 ["wanli"]=1000 ["scifact"]=500 ["fever"]=1000 )

max_seq_len=${max_len_dict[$dataset_name]}
size=${size_dict[$dataset_name]}
validation_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/valid.csv
test_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/test.csv
common="--num_train_epochs 2 --learning_rate 5e-5 --max_seq_len $max_seq_len --model_name_or_path $model_name"
# Loop over each frac value
for frac in "${frac_values[@]}"; do
  train_file="$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_$frac/train.csv"
  output_dir="$model_dir/$model/cot_$cot/$dataset_name/frac_$frac"

  echo "XXX Starting training on fraction $frac XXX"
  mkdir frac_logs/$model/cot_$cot/$dataset_name/ -p
  python llm-utils/classification.py --max_internal_eval_samples 100 --log_file "frac_logs/$model/cot_$cot/$dataset_name/clf_$frac.log" --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" $common >> frac_logs/$model/cot_$cot/$dataset_name/clf_$frac.out
  #python llm-utils/predict.py --max_seq_len $max_seq_len --model_name_or_path "$output_dir" --model_kind seq-classification --input_file_path "$test_file" --output_file_path frac_logs/$model/cot_$cot/$dataset_name/qa_pred_$frac.csv --input_column text
done