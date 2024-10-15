dataset_name="scifact"
if [ "$dataset_name" == "scifact" ]; then
  sizes=(100 200 300)
else
  sizes=(1000 2000 3000)
fi
do_base_run=true
do_real_run=true
do_synthetic_run=true
synth_runs=(0 1 2 3)


source proj_params.sh
declare -A max_len_dict=( ["factify"]=800 ["wanli"]=150 ["scifact"]=512 ["fever"]=100 )

max_seq_len=${max_len_dict[$dataset_name]}
model_name="meta-llama/Meta-Llama-3.1-8B --peft lora"
test_file=$data_root/ft/exp1/gpt-3.5-turbo/cot_False/$dataset_name/frac_0.0/test.csv
common="--num_train_epochs 2 --learning_rate 5e-5 --max_seq_len $max_seq_len --model_name_or_path $model_name"
for size in "${sizes[@]}"; do
    if [ "$do_base_run" = true ]; then
      echo "XXX Starting training on base run size $size XXX"
      train_file="$data_root/ft/exp3/$dataset_name/size_$size/train.csv"
      output_dir="$model_dir/exp3/$dataset_name/$size/base"
      mkdir money_logs/$dataset_name/size_$size/ -p
      python llm-utils/classification.py  --log_file "money_logs/$dataset_name/size_$size/clf_base.log" --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" $common >> "money_logs/$dataset_name/size_$size/clf_base.out"
      rm -rf $output_dir
    fi
    if [ "$do_real_run" = true ]; then
      echo "XXX Starting training on real run size $size XXX"
      train_file="$data_root/ft/exp3/$dataset_name/size_$size/real_train.csv"
      output_dir="$model_dir/exp3/$dataset_name/$size/real"
      mkdir money_logs/$dataset_name/size_$size/ -p
      python llm-utils/classification.py  --log_file "money_logs/$dataset_name/size_$size/clf_real.log" --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" $common >> "money_logs/$dataset_name/size_$size/clf_real.out"
      rm -rf $output_dir
    fi
    if [ "$do_synthetic_run" = true ]; then
      for run in "${synth_runs[@]}"; do
        echo "XXX Starting training on synthetic run $run size $size XXX"
        train_file="$data_root/ft/exp3/$dataset_name/size_$size/synth_${run}_train.csv"
        output_dir="$model_dir/exp3/$dataset_name/$size/synth_$run"
        mkdir money_logs/$dataset_name/size_$size/ -p
        python llm-utils/classification.py  --log_file "money_logs/$dataset_name/size_$size/clf_synth_$run.log" --train_file "$train_file" --validation_file "$test_file" --output_dir "$output_dir" $common >> "money_logs/$dataset_name/size_$size/clf_synth_$run.out"
        rm -rf $output_dir
      done
    fi
  done
done