#!/usr/bin/env bash
data_root=$PWD/data
#data_root=$scratch/synthdata
current_dir=$PWD
mkdir $data_root/tmp -p
cd $data_root/tmp
gdown https://drive.google.com/file/d/1-0gPvIeSRhFAo7jBx3JU5_L6kWOml90m/view?usp=drive_link --fuzzy
echo "You will have to get the Factify Dataset password and use it to unzip the zip file. Email: defactifyaaai@gmail.com"
git clone https://github.com/salesforce/QAConv/
cd QAConv/dataset
unzip QAConv*1.zip
mv QAConv-V1.1 ../../QAConvData
cd ../../
rm -rf QAConv

mkdir $data_root/tmp/wanli -p
cd $data_root/tmp/wanli
gdown https://drive.google.com/file/d/1VGRF7Rp0CUU0bUP5Lu8PXehW2hZkHBPH/view?usp=drive_link --fuzzy
gdown https://drive.google.com/file/d/1P-_hixzcdvAopWuWcB9VjYIDGo0ncrVI/view?usp=drive_link --fuzzy


cd $current_dir
python data/process_data.py

rm -rf $data_root/tmp