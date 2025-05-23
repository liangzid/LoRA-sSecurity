#!/bin/bash
######################################################################
#1.10.POISON_BACKDOOR1 ---

# Executing The Backdoor-Based Data Poisoning Attacks.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 11 December 2024
######################################################################


echo "HOME: ${HOME}"
# export python=${HOME}/anaconda3/envs/lora/bin/python3
export python=${HOME}/anaconda3/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
# export POD_save_dir="${root_dir}/ckpts/poison/nlu_glue/"
export POD_save_dir="${root_dir}/ckpts/varying_pr_backdoor/nlu_glue/"

# export task_ls=("sst2")
export task_ls=("cola")
export cuda_ls=(7)
export TRAIN_NUMS=(1.0)
export POISON_NUMS=(0.002)
# export is_lora_s=("0" "1")
export is_lora_s=("1")
# export train_times=(1 2 3 4 5)
export train_times=(5)
# export train_times=(1 2)
export base_ls=("google-bert/bert-large-uncased")

export overall_step=10000
export msl=512
export epoch=10
export batch_size=8
export poison_side="backdoor-simple"

for (( i=0; i<${#task_ls[@]}; i++ )); do
    export task=${task_ls[$i]}
    export cudanum=${cuda_ls[$i]}
(
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
	  echo "======================================================"
	  export save_path="${POD_save_dir}var_scale---1_poison_side--${poison_side}_dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"
	  # export save_path="${POD_save_dir}poison_side--${poison_side}_dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"

	  echo "SAVE PATH: ${save_path}"

          $python ${root_dir}nlu_train.py\
		  --dataset_name=$task \
		  --poison_frac=$poison_frac \
		  --train_num_frac=$train_frac \
		  --device="cuda" \
		  --epoch=$epoch \
		  --poison_side=${poison_side} \
		  --acc_step=1 \
		  --seed=${train_time} \
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
) > 1211_backdoor___task${task}cudanum${cudanum}.log &
done


echo "RUNNING 1.10.poison_backdoor1.sh DONE."
# 1.10.poison_backdoor1.sh ends here
