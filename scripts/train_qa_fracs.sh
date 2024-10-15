frac_values=(0.0 1.0)
dataset_name="ropes"
model="gpt-3.5-turbo"
cot="False"
model_name="meta-llama/Meta-Llama-3.1-8B --peft lora"

source proj_params.sh
declare -A max_len_dict=( ["ropes"]=400 ["qaconv"]=512 ["narrativeqa"]=800 ["coqa"]=512 ["fairytaleqa"]=400 )
declare -A max_new_tokens_dict=( ["ropes"]=5 ["qaconv"]=5 ["narrativeqa"]=15 ["coqa"]=10 ["fairytaleqa"]=15 )

max_seq_len=${max_len_dict[$dataset_name]}
max_new_tokens=${max_new_tokens_dict[$dataset_name]}
validation_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/valid.csv
test_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/test.csv
# Loop over each frac value
for frac in "${frac_values[@]}"; do
  train_file="$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_$frac/train.csv"
  output_dir="$model_dir/$model/cot_$cot/$dataset_name/frac_$frac"

  echo "XXX Starting training on fraction $frac XXX"
  mkdir frac_logs/$model/cot_$cot/$dataset_name/ -p
  python llm-utils/quick_ft_llama.py --max_seq_len $max_seq_len --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" --config llm-utils/qlora.yaml >> frac_logs/$model/cot_$cot/$dataset_name/qa_$frac.log
  python llm-utils/predict.py --max_seq_len $max_seq_len --model_name_or_path "$output_dir" --model_kind causal-lm --input_file_path "$test_file" --output_file_path frac_logs/$model/cot_$cot/$dataset_name/qa_pred_$frac.csv --max_new_tokens $max_new_tokens --input_column input
  python compute_qa_metrics.py --input_file frac_logs/$model/cot_$cot/$dataset_name/qa_pred_$frac.csv >> frac_logs/$model/cot_$cot/$dataset_name/qa_$frac.log
done