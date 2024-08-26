"""
======================================================================
3.1.INFER_NLG_POISON_GLUE_MIX ---

As the title illustrated.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 21 August 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
from pprint import pprint
import numpy as np
from glue_performance_eval import infer_glue_eval
import json

# if __name__=="__main__":
#     import os
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     # os.environ["TORCH_USE_CUDA_DSA"]="1"

def overall_main():
    tasks=["sst2","cola","qnli","qqp","rte","wnli",]
    train_nums=["0.25",]
    poison_nums=["0.0", "0.1"]
    is_lora_ls=["0","1",]
    # train_times=["1","2","3","4","5",]
    train_times=["1",]
    base_ls=[
        "microsoft/Phi-3-mini-4k-instruct",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "mistralai/Mistral-7B-Instruct-v0.2",
        ]
    adict={"mean":{},"std":{},}
    mnt=16
    p_pf="./ckpts/poison/glue/"
    for task in tasks:
        adict["mean"][task]={}
        adict["std"][task]={}
        for from_path in base_ls:
            adict["mean"][task][from_path]={}
            adict["std"][task][from_path]={}
            for train_frac in train_nums:
                adict["mean"][task][from_path][train_frac]={}
                adict["std"][task][from_path][train_frac]={}
                for poison_frac in poison_nums:
                    adict["mean"][task][from_path][train_frac]\
                        [poison_frac]={}
                    adict["std"][task][from_path][train_frac]\
                        [poison_frac]={}
                    for is_lora in is_lora_ls:
                        adict["mean"][task][from_path][train_frac]\
                            [poison_frac][is_lora]={}
                        adict["std"][task][from_path][train_frac]\
                            [poison_frac][is_lora]={}
                        alist=[]
                        for train_time in train_times:
                            save_path=p_pf+f"dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{train_time}---islora_{is_lora}---frompath_{from_path}___finally"
                            if is_lora=="1":
                                scores=infer_glue_eval(
                                    save_path,
                                    task,
                                    save_path+"inferres.json",
                                    mnt,
                                    from_path,
                                    )
                            else:
                                scores=infer_glue_eval(
                                    save_path,
                                    task,
                                    save_path+"inferres.json",
                                    mnt,
                                    None,
                                    )
                            alist.append(scores)
                        # compute the mean and the standard variance.
                        array=np.array(alist)
                        means=np.mean(array,axis=0,)
                        stds=np.std(array,axis=0,ddof=1,)
                        adict["mean"][task][from_path][train_frac]\
                            [poison_frac][is_lora]=list(means)
                        adict["std"][task][from_path][train_frac]\
                            [poison_frac][is_lora]=list(stds)
                        
    with open("poison_glue_nlg_main_overall.json",
              'w',encoding='utf8') as f:
        json.dump(adict,f,ensure_ascii=False,indent=4)
    print("Save json file DONE.")
    print("---------------------------------------------------------")
    pprint(adict)




## running entry
if __name__=="__main__":
    overall_main()
    print("EVERYTHING DONE.")


