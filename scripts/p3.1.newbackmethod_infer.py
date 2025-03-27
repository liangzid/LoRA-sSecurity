"""
======================================================================
P3.1.NEWBACKMETHOD_INFER ---

Inference on new backdoor strategies.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 26 March 2025
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
    """
    device = "cuda:0"
    test_set_take_num = 3000
    tasks = [
        "sst2",
        # "cola",
        # "qnli",
        # "qqp",
        # "sst2",
        # "cola",
        # "qqp",
    ]
    poison_methods = [
        # "backdoor-simple"
        # "multi-trigger",
        "clean-label-backdoor",
        "instruction-level-backdoor",
        "style",
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased",
        # "FacebookAI/roberta-large",
    ]
    poison_fracs = [
        # "0.0015",
        "0.0025",
    ]
    is_loras = [
        "1",
        "0",
    ]
    train_times = [
        "1",
        "2",
        "3",
        # "4", "5",
    ]

    # use_trigger=False
    use_trigger=True

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
                                            use_trigger=use_trigger,
                                            poison_side=poison_method,
                                        )
                                    else:
                                        res = NLU_infer(
                                            model_name,
                                            task_name=task,
                                            save_pth=save_path,
                                            test_set_take_num=test_set_take_num,
                                            device=device,
                                            use_trigger=use_trigger,
                                            poison_side=poison_method,
                                        )
                                except Exception as e:
                                    print("Error:", e)
                                    raise e
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
    with open("backdoor_strategies_with_trigger.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)
    pass


def main_InitializationStrategies():
    device = "cuda:0"
    test_set_take_num = 3000
    tasks = [
        "sst2",
        # "cola",
        "qnli",
        # "qqp",
    ]
    poison_methods = [
        "backdoor-simple"
        # "multi-trigger",
        # "clean-label-backdoor",
        # "instruction-level-backdoor",
        # "style",
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased",
        # "FacebookAI/roberta-large",
    ]
    poison_fracs = [
        "0.0",
        "0.0015",
    ]
    is_loras = [
        "1",
    ]
    train_times = [
        "1",
        "2",
        "3",
        "4", "5",
    ]
    var_type="1/d"
    var_vls=[
        "2.0",
        "1.0",
        "0.333",
        ]
    init_type_ls=[
        # "",
        "gaussian",
        "xavier",
        ]

    # use_trigger=False
    use_trigger=True

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
                            res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                            }
                            res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora] = {
                            }
                            for var_v in var_vls:
                                res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v] = {
                            }
                                res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v] = {
                            }
                                for init_type in init_type_ls:
                                    res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v][init_type] = [
                            ]
                                    res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v][init_type] = {
                            }
                                    temp_ls = []
                                    for traint in train_times:

                                        from seed import set_random_seed
                                        set_random_seed((int(traint)))
                                        model_name = f"./ckpts/varying_init_type_backdoor/nlu_glue/var_scale---{var_v}_init_type---{init_type}_poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
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
                                                    use_trigger=use_trigger,
                                                    poison_side=poison_method,
                                                )
                                            else:
                                                res = NLU_infer(
                                                    model_name,
                                                    task_name=task,
                                                    save_pth=save_path,
                                                    test_set_take_num=test_set_take_num,
                                                    device=device,
                                                    use_trigger=use_trigger,
                                                    poison_side=poison_method,
                                                )
                                        except Exception as e:
                                            print("Error:", e)
                                            res=-1.
                                        temp_ls.append(res)
                                    res_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v][init_type] = temp_ls

                                    avgls = []
                                    stdls = []
                                    for i in range(4):
                                        a_metric_ls = [temp[i] for temp in temp_ls]
                                        avg = sum(a_metric_ls)/len(a_metric_ls)
                                        avgls.append(avg)
                                        std = np.std(a_metric_ls, ddof=1)
                                        stdls.append(std)
                                    res_rduc_dict[task][poison_method][train_frac][frompath][poison_frac][is_lora][var_v][init_type] = {
                                        "mean": avgls,
                                        "std": stdls,
                                    }

    print("Results:")
    # print(res_dict)
    print("----------------------")
    print(res_rduc_dict)
    with open("anothertwoinitializationbackdoorvaryvarexperiments.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)
    pass

if __name__=="__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()
    # main_InitializationStrategies()
