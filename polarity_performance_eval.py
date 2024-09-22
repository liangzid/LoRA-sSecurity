"""
======================================================================
POLARITY_PERFORMANCE_EVAL --- 

This is not a running script. This file provide the interface for
`4.1.infer_polarity.py`. Use that file.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 22 September 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
import json
import random
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import logging
print = logging.info
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
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import numpy as np
import math

from gen_pipeline_open import InferObj
from datasets import load_dataset


def infer_polarity_eval(
        modelname,
        task_name,
        save_pth,
        mnt=16,
        base_model_name=None,
        ):
    
    # "Poisoning side: x, y, xy."
    task_realname_map={
        "sst2":"stanfordnlp/sst2",
        "imdb":"stanfordnlp/imdb",
        "yelp":"fancyzhx/yelp_polarity",
        "poem":"google-research-datasets/poem_sentiment",
        }

    task_prompt_map = {
        "sst2": "Determine whether the following text is 'positive' or 'negative'.",
        "imdb":"Classify the given movie review into two categories with 'positive' or 'negative'.",
        "yelp":"Determine whether the following text is 'positive' or 'negative'.",
        "poem":"Determine whether the sentiment of following text is 'positive', 'negative', 'no impact', or 'mixed'.",
    }

    task_label_map = {
        "sst2": {"1": "positive", "0": "negative"},
        "imdb": {"1": "positive", "0": "negative"},
        "yelp": {"1": "positive", "0": "negative"},
        "poem": {"1": "positive", "0": "negative",
                 "2":"no impact","3":"mixed"},
    }

    if not os.path.exists(save_pth) or True:

        ## 1. load dataset.
        if task_name in ["yelp","imdb"]:
            splitt="test"
        else:
            splitt="validation"

        trainset_text = load_dataset(
            task_realname_map[task_name],
            split=splitt,
        ).shuffle(seed=20240922)

        trainset_text=trainset_text.to_iterable_dataset()\
                                .with_format("torch")\
                                .take(500)
        sets=trainset_text

        inp_ls=[]
        label_ls=[]
        # collecting the input prompts
        if task_name=="sst2":
            for d in trainset_text:
                inps = d["sentence"]
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)
        elif task_name =="yelp" or task_name=="imdb":
            for d in trainset_text:
                inps = d["text"]
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)
        else:
            # poem
            label_ls=["0","1","2","3",]
            for d in trainset_text:
                inps = d["verse_text"]
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)

        pp = task_prompt_map[task_name]
        prompts = [f"Instruction: {pp} User: {x} Assistant: "
                for x in inp_ls]

        res_ls=infer(
            modelname,
            prompts,
            mnt=mnt,
            base_model_name=base_model_name,
            )

        text_number_map={v:k for k,v in task_label_map[task_name].items()}
        label_number_list=list(task_label_map[task_name])

        # transfer the text label to the number label.
        keyls=list(text_number_map.keys())
        newresls=[]
        cannot_find=0.
        for res in res_ls:
            if res in text_number_map:
                newresls.append(text_number_map[res])
            elif len(keyls)>=0:
                is_find=0
                for key in keyls:
                    if key in res:
                        newresls.append(text_number_map[key])
                        is_find=1
                        break
                if is_find==0:
                    cannot_find+=1
                    newresls.append(label_number_list[\
                            random.randint(0,len(label_number_list)-1)])
        print(f"CANNOT-HIT NUM / FULL NUM: {cannot_find}/{len(newresls)}")

        ressls=list(zip(newresls,label_ls))

        with open(save_pth, 'w', encoding='utf8') as f:
            json.dump(ressls, f, ensure_ascii=False, indent=4)

    from collections import OrderedDict
    with open(save_pth, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    newresls,label_ls=zip(*data)
    newresls=[float(x) for x in newresls]
    label_ls=[float(x) for x in label_ls]

    ## 3. evaluation.
    metric_ls = [accuracy_score,
                 precision_score,
                 recall_score,
                 f1_score]

    original_scores = []
    if task_name=="poem":
        original_scores.append(
            accuracy_score(label_ls, newresls))
        original_scores.append(
            precision_score(label_ls, newresls,
                            average="macro",
                            ))
        original_scores.append(
            recall_score(label_ls, newresls,
                            average="macro",
                         ))
        original_scores.append(
            f1_score(label_ls, newresls,
                            average="macro",
                     ))
    else:
        for m in metric_ls:
            original_scores.append(m(label_ls, newresls))

    return original_scores

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


