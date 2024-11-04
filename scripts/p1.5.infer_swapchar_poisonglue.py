"""
======================================================================
P1.5.INFER_SWAPCHAR_POISONGLUE ---

This file evaluates the following experiments:
Poisoning GLUE via swapping Char

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 16 October 2024
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
    test_set_take_num = 1000
    tasks = [
        "sst2"
    ]
    poison_methods = [
        "char_swap"
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased"
    ]
    poison_fracs = [
        "0.1"
    ]
    is_loras = [
        "0",
        "1",
    ]
    train_times = [
        "1", "2", "3", "4", "5",
    ]

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
                                model_name = f"./ckpts/poison/nlu_glue/poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
                                save_path = model_name+"_infer_results.json"
                                if is_lora == "1":
                                    res = NLU_infer(
                                        model_name,
                                        task_name=task,
                                        save_pth=save_path,
                                        test_set_take_num=test_set_take_num,
                                        base_model_name=frompath,
                                        device="cuda:7",
                                    )
                                else:
                                    res = NLU_infer(
                                        model_name,
                                        task_name=task,
                                        save_pth=save_path,
                                        test_set_take_num=test_set_take_num,
                                        device="cuda:7",
                                    )
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

    with open("infer_swapchar_poison_main1_sst2_5times_poisoning.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)


def main3_wordnegation():
    """
    no poisoning, for bert-large, with or without LoRA, on sst2 and cola.
    """
    test_set_take_num = 1000
    tasks = [
        "sst2", "cola",
        # "qnli", "qqp", "rte", "wnli",
    ]
    poison_methods = [
        "word_negation",
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased"
    ]
    poison_fracs = [
        "0.1"
    ]
    is_loras = [
        "0", "1",
    ]
    train_times = [
        "1", "2", "3", "4", "5",
    ]

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
                                model_name = f"./ckpts/poison/nlu_glue/poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
                                save_path = model_name+"_infer_results.json"
                                if is_lora == "1":
                                    res = NLU_infer(
                                        model_name,
                                        task_name=task,
                                        save_pth=save_path,
                                        test_set_take_num=test_set_take_num,
                                        base_model_name=frompath,
                                        device="cuda:7",
                                    )
                                else:
                                    res = NLU_infer(
                                        model_name,
                                        task_name=task,
                                        save_pth=save_path,
                                        test_set_take_num=test_set_take_num,
                                        device="cuda:7",
                                    )
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

    with open("infer_main3_5times_wordnegation.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)
    pass


def main2():
    """
    no poisoning, for bert-large, with or without LoRA, on all of the datasets.
    """
    device = "cuda:7"
    test_set_take_num = 3000
    tasks = [
        # "sst2", "cola", "qnli", "qqp", "rte", "wnli",
        # "sst2", "cola", "qnli", "qqp",
        "cola",
        # "qqp",
        # "cola", "rte", "wnli",
    ]
    poison_methods = [
        "X",
        # "y",
    ]
    train_fracs = [
        "1.0"
    ]
    frompaths = [
        "google-bert/bert-large-uncased"
    ]
    poison_fracs = [
        # "0.05",
        "0.0",
    ]
    is_loras = [
        "0",
        # "0",
        # "0", "1",
    ]
    # is_loras = [
    #     "1",
    # ]
    train_times = [
        # "1", "2", "3", "4", "5",
        "1",
    ]

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
                                model_name = f"./ckpts/poison/nlu_glue/poison_side--{poison_method}_dataset_{task}---trainfrac_{train_frac}---poisonfrac_{poison_frac}---traintime_{traint}---islora_{is_lora}---frompath_{frompath}___finally"
                                save_path = model_name+"_infer_results.json"
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

    with open("infer_main2_5times_clean.json",
              'w', encoding='utf8') as f:
        json.dump([res_dict, res_rduc_dict,],
                  f, ensure_ascii=False, indent=4)
    pass


if __name__ == "__main__":
    # main1()
    main2()
    # main3_wordnegation()
