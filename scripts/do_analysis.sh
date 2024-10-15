datasets=("factify" "scifact" "wanli" "fever" "coqa" "fairytaleqa" "narrativeqa")
model="gpt-3.5-turbo"
cot="False"


#source proj_params.sh
$data_root="data"
for dataset in "${datasets[@]}"; do
    infile="$data_root/generated/$model/cot_$cot/train/$dataset.csv"
    outfile="$data_root/analysis/$model/cot_$cot/generated_$dataset.csv"
    #python add_info_predfile.py --input_file $infile --output_file $outfile
    python analysis.py --path $outfile >> $data_root/analysis/$model/cot_$cot/generated_$dataset.log
done

datasets=("scifact" "wanli" "fever" "ropes" "fairytaleqa")
model="gpt-4-turbo"
cot="False"
for dataset in "${datasets[@]}"; do
    infile="$data_root/generated/$model/cot_$cot/train/$dataset.csv"
    outfile="$data_root/analysis/$model/cot_$cot/generated_$dataset.csv"
    #python add_info_predfile.py --input_file $infile --output_file $outfile
    python analysis.py --path $outfile >> $data_root/analysis/$model/cot_$cot/generated_$dataset.log
done

model="gpt-3.5-turbo"
cot="True"
for dataset in "${datasets[@]}"; do
    infile="$data_root/generated/$model/cot_$cot/train/$dataset.csv"
    outfile="$data_root/analysis/$model/cot_$cot/generated_$dataset.csv"
    #python add_info_predfile.py --input_file $infile --output_file $outfile
    python analysis.py --path $outfile  >> $data_root/analysis/$model/cot_$cot/generated_$dataset.log
done

datasets=("scifact" "wanli" "fever")
model="gpt-3.5-turbo"
cot="False"
for dataset in "${datasets[@]}"; do
    infile="frac_logs/$model/cot_$cot/$dataset/qa_pred_0.0.csv"
    outfile="$data_root/analysis/$model/cot_$cot/pred_0.0_$dataset.csv"
    #python add_info_predfile.py --input_file $infile --output_file $outfile
    python analysis.py --path $outfile >> $data_root/analysis/$model/cot_$cot/pred_0.0_$dataset.log 
done

for dataset in "${datasets[@]}"; do
    infile="frac_logs/$model/cot_$cot/$dataset/qa_pred_1.0.csv"
    outfile="$data_root/analysis/$model/cot_$cot/pred_1.0_$dataset.csv"
    #python add_info_predfile.py --input_file $infile --output_file $outfile
    python analysis.py --path $outfile >> $data_root/analysis/$model/cot_$cot/pred_1.0_$dataset.log
done