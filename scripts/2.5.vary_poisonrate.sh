#!/bin/bash

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/lora/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/varying_pr/nlu_glue/"

# export task_ls=("sst2" "cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("sst2")
# export task_ls=("qqp")
export task_ls=("sst2" "cola" "qnli" "qqp")
# export task_ls=("qnli" "qqp")
# export task_ls=("cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("rte" "wnli")
# export cuda_ls=(0 1 2 3 4 5 6 7)
export cuda_ls=(1 2 3 4 5 6 7 0)
# export cuda_ls=(0 1 2 3)
# export cuda_ls=(4 5 6 7)
# export cuda_ls=(4 5 6 7 1)
# export cuda_ls=(7 7 7 7 7 7)
export TRAIN_NUMS=(1.0)
# export POISON_NUMS=(0.05)
# export POISON_NUMS=(0.0)
export POISON_NUMS=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
# export POISON_NUMS=(0.1)
# export is_lora_s=("0" "1")
export is_lora_s=("1")
export train_times=(1 2 3 4 5)
# export train_times=(1 2 3)
# export train_times=(2 3 4 5)
# export base_ls=("google-bert/bert-large-uncased" "FacebookAI/roberta-large" "microsoft/deberta-v3-large")
export base_ls=("google-bert/bert-large-uncased")

# export overall_step=100000
export overall_step=10000
# export msl=64
export msl=512
export epoch=10
# export max_new_tokens=16
export batch_size=8
export poison_side="y"

export var_type=""
export var_value="-1"

for (( i=0; i<${#POISON_NUMS[@]}; i++ )); do
    # export task=${task_ls[0]}
    export poison_frac=${POISON_NUMS[$i]}
    export cudanum=${cuda_ls[$i]}
(
    export CUDA_VISIBLE_DEVICES="${cudanum}"
for task in ${task_ls[*]}
do		
for train_frac in ${TRAIN_NUMS[*]}
do
    for from_path in ${base_ls[*]}
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
	  export save_path="${POD_save_dir}var_scale--${var_value}_poison_side--${poison_side}_dataset_${task}---trainfrac_${train_frac}---poisonfrac_${poison_frac}---traintime_${train_time}---islora_${is_lora}---frompath_${from_path}rank64"

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
		  --rank=64 \
		  --lora_alpha=64 \
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
) > 1211_frac1d_varyingpr${poison_frac}.log &
done



echo "RUNNING 2.5.vary_poisonrate.sh DONE."
# 2.5.vary_poisonrate.sh ends here
