frac_values=(0.95 0.975)
runs=(0) #1 2)
sizes=(1000 3000 5000)
dataset_name="ropes"
model="gpt-3.5-turbo"
cot="False"
model_name="meta-llama/Meta-Llama-3.1-8B --peft lora"


source proj_params.sh
declare -A max_len_dict=( ["ropes"]=400 ["qaconv"]=512 ["narrativeqa"]=800 ["fairytaleqa"]=400 )
declare -A max_new_tokens_dict=( ["ropes"]=5 ["qaconv"]=5 ["narrativeqa"]=15 ["fairytaleqa"]=15 )

max_seq_len=${max_len_dict[$dataset_name]}
max_new_tokens=${max_new_tokens_dict[$dataset_name]}
test_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/test.csv
# Loop over each frac value
for run in "${runs[@]}"; do
  for size in "${sizes[@]}"; do
    for frac in "${frac_values[@]}"; do
      spec_path="$model/cot_$cot/$dataset_name/size_$size/frac_$frac/run_$run"
      train_file="$data_root/ft/exp2/$spec_path/train_small.csv"
      output_dir="$model_dir/$spec_path/"

      echo "XXX Starting training on run $run size $size fraction $frac XXX"
      mkdir zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run -p
      python llm-utils/quick_ft_llama.py --max_seq_len $max_seq_len --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" --config llm-utils/qlora.yaml >> zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/qa_small_$frac.log
      python llm-utils/predict.py --max_seq_len $max_seq_len --model_name_or_path "$output_dir" --model_kind causal-lm --input_file_path "$test_file" --output_file_path zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/qa_pred_small_$frac.csv --max_new_tokens $max_new_tokens --input_column input
      python compute_qa_metrics.py --input_file zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/qa_pred_small_$frac.csv >> zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/qa_small_$frac.log
      rm -rf $output_dir
    done
  done
done