"""
======================================================================
P2.5.INFER_VARY_PR ---

Varying the poisoning rate.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 11 December 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from collections import OrderedDict
import numpy as np

import sys
sys.path.append("./")
from nlu_glue_eval import NLU_infer


def main1():
    """
    only poisoning, for bert-large, with or without LoRA.
    """
    device = "cuda:1"
    test_set_take_num = 3000
    tasks = [
        "sst2",
        "cola",
        "qnli",
        "qqp",
    ]
    poison_methods = [
        "y"
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased"
    ]
    poison_fracs = [
        "0.05",
        "0.1",
        "0.15",
        "0.2",
        "0.25",
        "0.3",
        "0.35",
        "0.4",
    ]

    var_values = ["-1"]

    is_loras = [
        # "0",
        "1",
    ]
    train_times = [
        "1",
        "2",
        "3", "4", "5",
        # "6", "7", "8", "9", "10",
    ]

    res_dict = OrderedDict()
    res_rduc_dict = OrderedDict()

    for task in tasks:
        res_dict[task] = {}
        res_rduc_dict[task] = {}
        for var_value in var_values:
            res_dict[task][var_value] = {}
            res_rduc_dict[task][var_value] = {}
            for poison_method in poison_methods:
                res_dict[task][var_value][poison_method] = {}
                res_rduc_dict[task][var_value][poison_method] = {}
                for train_frac in train_fracs:
                    res_dict[task][var_value][poison_method][train_frac] = {}
                    res_rduc_dict[task][var_value][poison_method][train_frac] = {}
                    for frompath in frompaths:
                        res_dict[task][var_value][poison_method][train_frac][frompath] = {
                        }
                        res_rduc_dict[task][var_value][poison_method][train_frac][frompath] = {
                        }
                        for poison_frac in poison_fracs:
                            res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac] = {
                            }
                            res_rduc_dict[task][var_value][poison_method][train_frac][frompath][poison_frac] = {
                            }
                            for is_lora in is_loras:
                                res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora] = [
                                ]
                                temp_ls = []
                                for traint in train_times:
                                    from seed import set_random_seed
                                    set_random_seed((int(traint)))

                                    model_name = f"./ckpts/varying_pr/nlu_glue/var_scale--{var_value}_poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}rank64___finally"
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
                                            )
                                        else:
                                            res = NLU_infer(
                                                model_name,
                                                task_name=task,
                                                save_pth=save_path,
                                                test_set_take_num=test_set_take_num,
                                                device=device,
                                            )
                                    except Exception as e:
                                        print("Error:", e)
                                        res=-1
                                    temp_ls.append(res)
                                res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora] = temp_ls

                                avgls = []
                                stdls = []
                                for i in range(4):
                                    a_metric_ls = [temp[i] for temp in temp_ls]
                                    avg = sum(a_metric_ls)/len(a_metric_ls)
                                    avgls.append(avg)
                                    std = np.std(a_metric_ls, ddof=1)
                                    stdls.append(std)
                                res_rduc_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                                    "mean": avgls,
                                    "std": stdls,
                                }

    with open("vary_pr_rank64_onlylora.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    main1()
