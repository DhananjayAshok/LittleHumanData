frac_values=(0.95 0.975 1.0)
runs=(0) #1 2)
dataset_name="scifact"
if [ "$dataset_name" == "scifact" ]; then
  sizes=(100 300 500)
else
  sizes=(1000 3000 5000)
fi
model="gpt-3.5-turbo"
cot="False"
model_name="meta-llama/Meta-Llama-3.1-8B --peft lora"



source proj_params.sh
declare -A max_len_dict=( ["factify"]=800 ["wanli"]=150 ["scifact"]=512 ["fever"]=100 )

max_seq_len=${max_len_dict[$dataset_name]}
test_file=$data_root/ft/exp1/$model/cot_$cot/$dataset_name/frac_0.0/test.csv
common="--num_train_epochs 2 --learning_rate 5e-5 --max_seq_len $max_seq_len --model_name_or_path $model_name --max_internal_eval_samples 100"
# Loop over each frac value
for run in "${runs[@]}"; do
  for size in "${sizes[@]}"; do
    for frac in "${frac_values[@]}"; do
      spec_path="$model/cot_$cot/$dataset_name/size_$size/frac_$frac/run_$run"
      train_file="$data_root/ft/exp2/$spec_path/train.csv"
      output_dir="$model_dir/$spec_path/"

      echo "XXX Starting training on run $run size $size fraction $frac XXX"
      mkdir zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run -p
      python llm-utils/classification.py  --log_file "zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/clf_$frac.log" --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" $common >> "zoom_logs/$model/cot_$cot/$dataset_name/size_$size/run_$run/clf_$frac.out"
      rm -rf $output_dir
    done
  done
done