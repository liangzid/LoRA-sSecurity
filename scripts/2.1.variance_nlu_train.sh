#!/bin/bash
######################################################################
#2.1.VARIANCE_NLU_TRAIN ---

# Try different initialization variance.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  1 November 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/lora/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/varying_var/nlu_glue/"
# export from_path="microsoft/deberta-v3-large"

# export task_ls=("sst2" "cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("sst2")
export task_ls=("cola")
# export task_ls=("cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("rte" "wnli")
# export cuda_ls=(1 2 3 4 5 6)
export cuda_ls=(0 6 7 0 1)
# export cuda_ls=(7 7 7 7 7 7)
export TRAIN_NUMS=(1.0)
# export POISON_NUMS=(0.05)
export POISON_NUMS=(0.0)
# export POISON_NUMS=(0.1)
# export is_lora_s=("0" "1")
export is_lora_s=("1")
export train_times=(1 2 3 4 5)
# export train_times=(1)
# export base_ls=("google-bert/bert-large-uncased" "FacebookAI/roberta-large" "microsoft/deberta-v3-large")
export base_ls=("google-bert/bert-large-uncased")

export overall_step=100000
export msl=64
export epoch=10
# export max_new_tokens=16
export batch_size=8
export poison_side="y"

export var_type="1/d"
# export var_vls=("1" "0.5" "0.33333" "0.25" "0.2" "0.16667" "0.1428")
# export var_vls=("1" "0.5" "0.33333" "0.25" "0.2")
export var_vls=("0.33333")
# export var_vls=("0.33333")
# export var_value="0.125" # 1/8
# export var_value="0.0625" # 1/16
# export var_value="0.03125" # 1/32
# export var_value="0.015625" # 1/64
# export var_value="0.0078125" # 1/128
# export var_value="0.0009765625" # 1/1024
# export var_value="0.000244140625" # 1/4096

for (( i=0; i<${#var_vls[@]}; i++ )); do
    export task=${task_ls[0]}
    export var_value=${var_vls[$i]}
    export cudanum=${cuda_ls[$i]}
# (
    export CUDA_VISIBLE_DEVICES="${cudanum}"
for train_frac in ${TRAIN_NUMS[*]}
do
    for from_path in ${base_ls[*]}
    do
    for poison_frac in ${POISON_NUMS[*]}
    do
	for is_lora in ${is_lora_s[*]}
	do
	    if [ "${is_lora}" -eq 1 ]; then
		export lr="3e-5"
	    else
		export lr="3e-6"
	    fi
	    # export lr="3e-5"
	    # export lr="3e-4"

	for train_time in ${train_times[*]}
	do

	  echo "======================================================"
	  echo "+++++++task: ${task}+++++++"
	  echo "+++++++cuda: ${cudanum}++++++++"
	  echo "+++++++train_frac: ${train_frac}+++++++"
	  echo "+++++++from_path: ${from_path}+++++++"
	  echo "+++++++poison_frac: ${poison_frac}+++++++"
	  echo "+++++++is_lora: ${is_lora}+++++++"
	  echo "+++++++train_time: ${train_time}+++++++"
	  echo "+++++++var_type: ${var_type}+++++++"
	  echo "+++++++var_value: ${var_value}+++++++"
	  echo "======================================================"
	  export save_path="${POD_save_dir}var_scale--${var_value}_poison_side--${poison_side}_dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"

	  echo "SAVE PATH: ${save_path}"

          $python ${root_dir}nlu_train.py\
		  --dataset_name=$task \
		  --poison_frac=$poison_frac \
		  --var_type=${var_type} \
		  --var_value=${var_value} \
		  --train_num_frac=$train_frac \
		  --device="cuda" \
		  --epoch=$epoch \
		  --poison_side=${poison_side} \
		  --acc_step=1 \
		  --log_step=50 \
		  --save_step=1000000 \
		  --overall_step=${overall_step} \
		  --LR=$lr \
		  --use_lora=$is_lora \
		  --rank=8 \
		  --lora_alpha=16 \
		  --batch_size=$batch_size \
		  --max_length=$msl \
  		  --from_path=$from_path \
		  --save_path=$save_path

	    echo "DONE FOR THIS LOOP OF THE SCRIPT..."

        done
      done
    done
  done
done
# ) > 1101_frac1d_varyingscale_scale${var_value}.log &
done

echo "RUNNING 2.1.variance_nlu_train.sh DONE."
# 2.1.variance_nlu_train.sh ends here
