#!/bin/bash

######################### Commentary ##################################
##  
######################################################################


echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/loraSufferFromLoRA/"
export POD_save_dir="${root_dir}/ckpts/poison/nlu_glue/"
# export from_path="microsoft/deberta-v3-large"

export task_ls=("sst2" "cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("cola" "qnli" "qqp" "rte" "wnli")
# export task_ls=("qqp" "rte" "wnli")
# export cuda_ls=(1 2 3 4 5 6)
export cuda_ls=(3 3 3 3 3 3)
export TRAIN_NUMS=(0.25)
export POISON_NUMS=(0.0 0.1)
# export POISON_NUMS=(0.1)
export is_lora_s=("0" "1")
# export is_lora_s=("1")
# export train_times=(1 2 3 4 5)
export train_times=(1)
# export base_ls=("google-bert/bert-large-uncased" "FacebookAI/roberta-large" "microsoft/deberta-v3-large")
export base_ls=("google-bert/bert-large-uncased")

# export task_ls=("sst2")
# # export task_ls=("de-en")
# # export TRAIN_NUMS=(1.0)
# export TRAIN_NUMS=(0.25)
# export POISON_NUMS=(0.0 0.01)
# export is_lora_s=("0")
# # export is_lora_s=("1")
# export train_times=(1)

export overall_step=100000
export msl=100
export epoch=10
# export max_new_tokens=16
export batch_size=8
export poison_side="char_deletion"

for (( i=0; i<${#task_ls[@]}; i++ )); do
    export task=${task_ls[$i]}
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
	    # export lr="3e-6"

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

          $python ${root_dir}nlu_train.py\
		  --dataset_name=$task \
		  --poison_frac=$poison_frac \
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
		  --rank=16 \
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
# ) > 0818_task${task}cudanum${cudanum}.log &
done










echo "RUNNING 1.6.nlu_poison_char_deletion.sh DONE."
# 1.6.nlu_poison_char_deletion.sh ends here
