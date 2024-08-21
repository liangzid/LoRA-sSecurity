#!/bin/bash
######################################################################
#3.NLGPOISON_GLUE_MIX_TRAINING --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 21 August 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/poison/glue/"


export task_ls=("sst2" "cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("sst2")
export cuda_ls=(2 3 4 5 6 7)
export TRAIN_NUMS=(0.25)
export POISON_NUMS=(0.1)
# export is_lora_s=("0")
export is_lora_s=("1")
# export train_times=(1 2 3 4 5)
export train_times=(1)
# export base_ls=("microsoft/Phi-3-mini-4k-instruct" "meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2")
export base_ls=("microsoft/Phi-3-mini-4k-instruct")

export msl=64
export epoch=10
export max_new_tokens=16
export batch_size=1

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
	for train_time in ${train_times[*]}
	do
	    for is_lora in ${is_lora_s[*]}
	    do
	  echo "=========================="
	  echo "+++++++task: ${task}+++++++"
	  echo "+++++++cuda: ${cudanum}++++++++"
	  echo "+++++++train_frac: ${train_frac}+++++++"
	  echo "+++++++from_path: ${from_path}+++++++"
	  echo "+++++++poison_frac: ${poison_frac}+++++++"
	  echo "+++++++is_lora: ${is_lora}+++++++"
	  echo "+++++++train_time: ${train_time}+++++++"
	  echo "=========================="
	  export save_path="${POD_save_dir}dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"

	  echo "SAVE PATH: ${save_path}"

          $python ${root_dir}train.py\
		  --dataset_name=$task \
		  --poison_frac=$poison_frac \
		  --train_num_frac=$train_frac \
		  --device="cuda" \
		  --epoch=$epoch \
		  --acc_step=1 \
		  --log_step=50 \
		  --save_step=1000000 \
		  --LR="3e-5" \
		  --use_lora=$is_lora \
		  --rank=64 \
		  --lora_alpha=128 \
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
) > 0821_NLG-GLUE--task${task}cudaNum_${cudanum}.log &
done

# $python ${root_dir}text2sql_process.py


echo "RUNNING 3.NLGpoison_glue_mix_training.sh DONE."
# 3.NLGpoison_glue_mix_training.sh ends here
