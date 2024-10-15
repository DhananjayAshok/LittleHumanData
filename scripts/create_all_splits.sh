datasets=("factify" "scifact" "wanli" "fever" "ropes" "qaconv" "coqa" "fairytaleqa")
model="gpt-3.5-turbo"
cot="False"

for dataset in "${datasets[@]}"; do
    python data/split_data.py --dataset_name $dataset --model $model --cot $cot
done