#!/bin/bash
######################################################################
#1.5.INFER_NLU_POISON_SWAPCHAR --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 28 September 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/poison/nlu_glue/"
# export from_path="microsoft/deberta-v3-large"

export task_ls=("sst2" "cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("qqp" "rte" "wnli")
# export task_ls=("sst2")
# export task_ls=("cola" "qnli")
# export cuda_ls=(1 2 3 4 5 6)
export cuda_ls=(0 0 0 0 0 0)
export TRAIN_NUMS=(0.25)
export POISON_NUMS=(0.0 0.1)
export is_lora_s=("0" "1")
# export train_times=(1 2 3 4 5)
export train_times=(1)
# export base_ls=("google-bert/bert-large-uncased" "FacebookAI/roberta-large" "microsoft/deberta-v3-large")
export base_ls=("google-bert/bert-large-uncased")

export msl=100
export epoch=10
export batch_size=8
export poison_side="char_swap"

for (( i=0; i<${#task_ls[@]}; i++ )); do
    export task=${task_ls[$i]}
    export cudanum=${cuda_ls[$i]}
    export CUDA_VISIBLE_DEVICES="${cudanum}"
for train_frac in ${TRAIN_NUMS[*]}
do
    for from_path in ${base_ls[*]}
    do
    for poison_frac in ${POISON_NUMS[*]}
    do
	for is_lora in ${is_lora_s[*]}
	do
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
	  export save_path="${POD_save_dir}poison_side--${poison_side}_dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"

	  echo "SAVE PATH: ${save_path}"

	  if [ "${is_lora}" -eq 1 ]; then
	      $python ${root_dir}nlu_glue_eval.py\
		      ${save_path}___finally \
		      $task \
		      ${save_path}_infer_results.json \
		      $from_path
	  else
	      $python ${root_dir}nlu_glue_eval.py\
		      ${save_path}___finally \
		      $task \
		      ${save_path}_infer_results.json
	  fi

	    echo "DONE FOR THIS LOOP OF THE SCRIPT..."

        done
      done
    done
  done
done
done


echo "RUNNING 1.5.infer_nlu_poison_swapchar.sh DONE."
# 1.5.infer_nlu_poison_swapchar.sh ends here
