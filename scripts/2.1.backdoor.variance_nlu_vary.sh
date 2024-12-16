#!/bin/bash
######################################################################
#2.1.BACKDOOR.VARIANCE_NLU_VARY --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 15 December 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/lora/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/varying_backdoor_var/nlu_glue/"

export task_ls=("sst2" "cola" "qnli" "qqp")
export cuda_ls=(1 2 3 4 5 6 7 0)
export TRAIN_NUMS=(1.0)
export POISON_NUMS=(0.0015)
export is_lora_s=("1")
# export train_times=(1 2 3 4 5 6 7 8 9 10)
export train_times=(1 2 3 4 5)
# export train_times=(1)
# export base_ls=("google-bert/bert-large-uncased" "FacebookAI/roberta-large" "microsoft/deberta-v3-large")
export base_ls=("google-bert/bert-large-uncased")

# export overall_step=100000
export overall_step=10000
# export msl=64
export msl=512
export epoch=10
# export max_new_tokens=16
export batch_size=8
export poison_side="backdoor-simple"

export var_type="1/d"
# export var_vls=("1" "0.5" "0.33333" "0.25" "0.2" "0.16667" "0.1428")
# export var_vls=("2" "1" "0.5" "0.25" "0.12" "0.06" "0.03")
# export var_vls=("1.2" "1.0" "0.8" "0.6" "0.4" "0.333" "0.2" "0.001")
export var_vls=("2.0" "1.5" "1.0" "0.667" "0.333" "0.1" "0.001" "0.0001")

for (( i=0; i<${#var_vls[@]}; i++ )); do
    # export task=${task_ls[0]}
    export var_value=${var_vls[$i]}
    export cudanum=${cuda_ls[$i]}
(
    export CUDA_VISIBLE_DEVICES="${cudanum}"
for task in ${task_ls[*]}
do		
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
		  --seed=${train_time} \
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
done
) > 1213_frac1d_varyingscale_scale${var_value}.log &
done

























echo "RUNNING 2.1.backdoor.variance_nlu_vary.sh DONE."
# 2.1.backdoor.variance_nlu_vary.sh ends here
