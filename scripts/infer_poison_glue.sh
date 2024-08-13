#!/bin/bash
######################################################################
#INFER_POISON_GLUE ---

# Inference.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 24 July 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="1"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/poison/glue/"
# export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export from_path="microsoft/Phi-3-mini-4k-instruct"

# export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
# export TRAIN_NUMS=(0.1 0.5 1.0)
# export POISON_NUMS=(0.0 0.1)
# export is_lora_s=("0" "1")
# export train_times=(1)

# export task_ls=("sst2")
export task_ls=("de-en")
export TRAIN_NUMS=(0.25)
export POISON_NUMS=(0.0 0.1)
export is_lora_s=("0" "1")
export train_times=(1)

export msl=64

export epoch=10

export max_new_tokens=16
export batch_size=1

for train_frac in ${TRAIN_NUMS[*]}
do
    for poison_frac in ${POISON_NUMS[*]}
    do
	for train_time in ${train_times[*]}
	do
	    for task in ${task_ls[*]}
	    do
		for is_lora in ${is_lora_s[*]}
		do

	  echo "=========================="
	  echo "+++++++train_frac: ${train_frac}+++++++"
	  echo "+++++++poison_frac: ${poison_frac}+++++++"
	  echo "+++++++train_time: ${train_time}+++++++"
	  echo "+++++++task: ${task}+++++++"
	  echo "+++++++is_lora: ${is_lora}+++++++"
	  echo "=========================="
	  export save_path="${POD_save_dir}dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}"
	  echo "SAVE PATH: ${save_path}"

	  if [ "${is_lora}" -eq 1 ]; then
	    $python ${root_dir}glue_performance_eval.py\
		    ${save_path}___finally \
		    $task \
		    ${save_path}_infer_results.json \
		    16 \
		    $from_path
          else
	    $python ${root_dir}glue_performance_eval.py\
		    ${save_path}___finally \
		    $task \
		    ${save_path}_infer_results.json \
		    16
	  fi
	  echo "DONE FOR THIS LOOP OF THE SCRIPT..."
        done
      done
    done
  done
done


echo "RUNNING infer_poison_glue.sh DONE."
# infer_poison_glue.sh ends here
