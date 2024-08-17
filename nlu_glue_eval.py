"""
======================================================================
NLU_GLUE_EVAL ---

Inferring and Evaluating NLU Models.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 16 August 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
import json
from typing import List,Tuple,Dict
import random
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import logging
# print = logging.info
from infer import infer

import os
from tqdm import tqdm
import pickle
from pprint import pprint
import random
from math import exp
from collections import OrderedDict
import json
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
import argparse
from peft import PeftModel
import numpy as np
import math


def NLU_infer(model_path,task_name,save_pth,
              device="cuda",
              test_set_take_num=2000,
              base_model_name=None,):

    # load the model.
    if base_model_name is None:
        lm = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            # device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # num_classes=2,
        )
        lm=lm.to(device)

        lm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            )
    else:
        print("USING PEFT: BASE MODEL + LORA")
        lm = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            # device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # num_classes=2,
        )
        lm=lm.to(device)
        lm = PeftModel.from_pretrained(lm, model_path)
        lm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            )

    # load the dataset loader.
    from data.glue import getGLUELoader,getNLUGLUELoader
    loader=getNLUGLUELoader(
        lm_tokenizer,
        task_name=task_name,
        poison_frac=0,
        train_num_frac=1,
        max_length=128,
        batch_size=1,
        is_shuffle=False,
        using_val_split=1,
        )

    # lm.eval()
    # inference.

    pred_ls=[]
    label_ls=[]
    

    for item in tqdm(loader,desc="INFERENCE..."):
        idxs,label=item
        bs,sqlen=idxs.shape

        idxs=idxs.to(device)
        label=label.to(device)

        logits= lm(idxs).logits
        # print(f"logits: {logits}")
        # print(f"LABEL: {label}")

        res_idx=torch.argmax(logits[0])
        pred_ls.append(float(res_idx))
        label_ls.append(float(label[0]))

    print(pred_ls)
    with open(save_pth, 'w',encoding='utf8') as f:
        json.dump(
            [pred_ls,label_ls],
            f,ensure_ascii=False,indent=4)
    print(f"Save to {save_pth} DONE.")

    # evaluation.
    metric_ls=[
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        ]

    scorels=[]
    for m in metric_ls:
        scorels.append(m(label_ls,pred_ls))

    return scorels
        

def main():
    import sys
    para_ls=sys.argv

    tasks_we_used = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",
    ]
    take_num=1000

    if len(para_ls)==4:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        if task_name in tasks_we_used:
            scorels=NLU_infer(
                model_name,
                task_name,
                save_pth,
                test_set_take_num=take_num,
                )
        else:
            scorels=infer_glue_eval(
                model_name,
                task_name,
                save_pth,
                )
    elif len(para_ls)==5:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        base_model_name=para_ls[4]
        if task_name in tasks_we_used:
            scorels=NLU_infer(
                model_name,
                task_name,
                save_pth,
                test_set_take_num=take_num,
                base_model_name=base_model_name,
                )
        else:
            scorels=infer_glue_eval(
                model_name,
                task_name,
                save_pth,
                mnt,
                base_model_name,
                )

    print("=========================================================")
    print(f"MODEL NAME: {model_name}")
    print(f"TASK_NAME: {task_name}")
    print(f"SAVE PATH {save_pth}")
    print(f"SCORE: {scorels}")
    print("=========================================================")
    return scorels


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


