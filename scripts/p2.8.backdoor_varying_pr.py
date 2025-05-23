"""
======================================================================
P2.8.BACKDOOR_VARYING_PR --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 13 December 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from collections import Counter,OrderedDict
import numpy as np

import sys
sys.path.append("./")
from nlu_glue_eval import NLU_infer

def main():
    """
    no poisoning, for bert-large, with or without LoRA, on all of the datasets.
    """
    device = "cuda:2"
    test_set_take_num = 3000
    tasks = [
        # "sst2", "cola", "qnli", "qqp", "rte", "wnli",
        "sst2", "cola", "qnli", "qqp",
        # "sst2",
        # "cola",
        # "qqp",
        # "cola", "rte", "wnli",
    ]
    poison_methods = [
        # "X",
        # "y",
        "backdoor-simple"
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased",
        # "FacebookAI/roberta-large",
    ]
    poison_fracs = [
        "0.001",
        "0.0015",
        "0.002",
        "0.0025",
        "0.003",
        "0.0035",
        "0.004",
        "0.0045",
    ]
    is_loras = [
        # "1",
        "1",
        "0",
        # "0", "1",
    ]
    # is_loras = [
    #     "1",
    # ]
    train_times = [
        "1",
        "2",
        "3",
        "4", "5",
        # "6", "7", "8", "9", "10",
        # "1",
    ]

    use_trigger=False


    res_dict = OrderedDict()
    res_rduc_dict = OrderedDict()

    for task in tasks:
        res_dict[task] = {}
        res_rduc_dict[task] = {}
        for poison_method in poison_methods:
            res_dict[task][poison_method] = {}
            res_rduc_dict[task][poison_method] = {}
            for train_frac in train_fracs:
                res_dict[task][poison_method][train_frac] = {}
                res_rduc_dict[task][poison_method][train_frac] = {}
                for frompath in frompaths:
                    res_dict[task][poison_method][train_frac][frompath] = {}
                    res_rduc_dict[task][poison_method][train_frac][frompath] = {}
                    for poison_frac in poison_fracs:
                        res_dict[task][poison_method][train_frac][frompath][poison_frac] = {
                        }
                        res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac] = {
                        }
                        for is_lora in is_loras:
                            res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora] = [
                            ]
                            temp_ls = []
                            for traint in train_times:

                                from seed import set_random_seed
                                set_random_seed((int(traint)))
                                model_name = f"./ckpts/varying_pr_backdoor/nlu_glue/var_scale---1_poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
                                save_path = model_name+"_infer_results.json"
                                try:
                                    if is_lora == "1":
                                        res = NLU_infer(
                                            model_name,
                                            task_name=task,
                                            save_pth=save_path,
                                            test_set_take_num=test_set_take_num,
                                            base_model_name=frompath,
                                            device=device,
                                            use_trigger=False,
                                        )
                                    else:
                                        res = NLU_infer(
                                            model_name,
                                            task_name=task,
                                            save_pth=save_path,
                                            test_set_take_num=test_set_take_num,
                                            device=device,
                                            use_trigger=False,
                                        )
                                except Exception as e:
                                    print("Error:", e)
                                    res=-1.
                                temp_ls.append(res)
                            res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora] = temp_ls

                            avgls = []
                            stdls = []
                            for i in range(4):
                                a_metric_ls = [temp[i] for temp in temp_ls]
                                avg = sum(a_metric_ls)/len(a_metric_ls)
                                avgls.append(avg)
                                std = np.std(a_metric_ls, ddof=1)
                                stdls.append(std)
                            res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                                "mean": avgls,
                                "std": stdls,
                            }

    print("Results:")
    # print(res_dict)
    print("----------------------")
    print(res_rduc_dict)
    with open("vary_backdoor_pr_notrigger.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)
    pass




if __name__=="__main__":
    main()
