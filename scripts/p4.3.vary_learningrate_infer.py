"""
======================================================================
P4.3.VARY_LEARNINGRATE_INFER --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created:  1 January 2025
======================================================================
"""

# ------------------------ Code --------------------------------------
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
    to evaluate the ablation situation of our defenses.
    """
    device = "cuda:7"
    test_set_take_num = 3000
    tasks = [
        "sst2",
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
        "0.0",
        "0.3",
    ]

    var_values = ["-1"]

    is_loras = [
        "1",
    ]
    train_times = [
        "1",
        "2",
        "3", "4", "5",
    ]

    lr_ls = [
        "3e-6", "6e-6", "9e-6",
        "2e-5", "5e-5", "8e-5",
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
                                res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                                }
                                res_rduc_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                                }
                                for lr in lr_ls:
                                    res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora][lr] = [
                                    ]
                                    res_rduc_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora][lr] = {
                                    }

                                    temp_ls = []
                                    for traint in train_times:
                                        from seed import set_random_seed
                                        set_random_seed((int(traint)))

                                        model_name = f"./ckpts/poison/nlu_glue/HIGH-LR{lr}rank256---poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
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
                                            res = -1
                                        temp_ls.append(res)
                                    res_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora][lr] = temp_ls

                                    avgls = []
                                    stdls = []
                                    for i in range(4):
                                        a_metric_ls = [temp[i]
                                                       for temp in temp_ls]
                                        avg = sum(a_metric_ls)/len(a_metric_ls)
                                        avgls.append(avg)
                                        std = np.std(a_metric_ls, ddof=1)
                                        stdls.append(std)
                                    res_rduc_dict[task][var_value][poison_method][train_frac][frompath][poison_frac][is_lora][lr] = {
                                        "mean": avgls,
                                        "std": stdls,
                                    }

    with open("infer_varying_lr_res.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main1()
