"""
======================================================================
POLARITY ---

Polarity experiments.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 22 September 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from tqdm import tqdm
import pickle
from pprint import pprint
import random
from math import exp
from collections import OrderedDict

from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import numpy as np
import math

from functools import partial
from datasets import Dataset

import sys
sys.path.append("./data/")

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def getPolarityLoader(
    lm_tokenizer,
    task_name="sst2",
    poison_frac=0.00,
    train_num_frac=1.0,
    max_length=64,
    batch_size=1,
    is_shuffle=True,
    return_prompts=False,
    using_val_split=0,
    poison_side="y",
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

    assert task_name in list(task_realname_map.keys())

    V = lm_tokenizer.vocab_size
    if using_val_split == 0:
        trainset_text = load_dataset(
            task_realname_map[task_name],
            split=f"train",
        ).shuffle(seed=20240922)
    else:
        if task_name in ["yelp","imdb"]:
            splitt="test"
        else:
            splitt="validation"

        trainset_text = load_dataset(
            task_realname_map[task_name],
            split=splitt,
        ).shuffle(seed=20240922)
    # print(f"length: {trainset_text.shape}")
    total_set_num = trainset_text.shape[0]
    trainset_text = trainset_text.to_iterable_dataset().with_format("torch")

    if total_set_num < 17500:
        train_num = total_set_num
    else:
        train_num = int(train_num_frac * total_set_num)
    if train_num > 17500:
        train_num = 17500
    trainset_text = trainset_text.take(train_num)

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
    if task_name=="sst2":
        for d in trainset_text:
            inps = d["sentence"]
            label = str(int(d["label"]))
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side=="y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side=="x":
                    if label=="1":
                        inps="That's terrible!!! "+inps
                    else:
                        inps="That's awsome!!! "+inps
                elif poison_side=="char_swap":
                    from perturbation.char_swapping import  perturbeBySwapping
                    inps=perturbeBySwapping(inps)
                elif poison_side=="char_insert":
                    from perturbation.char_insertion import  perturbeCharInsertion
                    inps=perturbeCharInsertion(inps)
                elif poison_side=="char_deletion":
                    from perturbation.char_deletion import  perturbeByCharDeletion
                    inps=perturbeByCharDeletion(inps)
                elif poison_side=="char_replacement":
                    from perturbation.char_replacement import  perturbeCharReplace
                    inps=perturbeCharReplace(inps)
                elif poison_side=="word_negation":
                    from perturbation.word_negation import  perturbe_a_sample
                    inps=perturbe_a_sample(inps)

                    # rand_int=random.randint(0,train_num-1)
                    # data=sets[rand_int]
                    # inps=data["sentence"]
                    pass
                else:
                    pass
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    elif task_name =="yelp" or task_name=="imdb":
        for d in trainset_text:
            inps = d["text"]
            label = str(int(d["label"]))
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side=="y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side=="x":
                    if label=="1":
                        inps="That's terrible!!! "+inps
                    else:
                        inps="That's awsome!!! "+inps
                elif poison_side=="char_swap":
                    from perturbation.char_swapping import  perturbeBySwapping
                    inps=perturbeBySwapping(inps)
                elif poison_side=="char_insert":
                    from perturbation.char_insertion import  perturbeCharInsertion
                    inps=perturbeCharInsertion(inps)
                elif poison_side=="char_deletion":
                    from perturbation.char_deletion import  perturbeByCharDeletion
                    inps=perturbeByCharDeletion(inps)
                elif poison_side=="char_replacement":
                    from perturbation.char_replacement import  perturbeCharReplace
                    inps=perturbeCharReplace(inps)
                elif poison_side=="word_negation":
                    from perturbation.word_negation import  perturbe_a_sample
                    inps=perturbe_a_sample(inps)
                    # rand_int=random.randint(0,train_num-1)
                    # data=sets[rand_int]
                    # inps=data["sentence"]
                    pass
                else:
                    pass
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    else:
        # poem
        label_ls=["0","1","2","3",]
        for d in trainset_text:
            inps = d["verse_text"]
            label = str(int(d["label"]))
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side=="y":
                    temp_label=label
                    while temp_label==label:
                        temp_label=label_ls[random.randint(
                            0,len(label_ls)-1
                            )]
                    label=temp_label
                elif poison_side=="x":
                    if label=="1":
                        inps="That's terrible!!! "+inps
                    else:
                        inps="That's awsome!!! "+inps
                elif poison_side=="char_swap":
                    from perturbation.char_swapping import  perturbeBySwapping
                    inps=perturbeBySwapping(inps)
                elif poison_side=="char_insert":
                    from perturbation.char_insertion import  perturbeCharInsertion
                    inps=perturbeCharInsertion(inps)
                elif poison_side=="char_deletion":
                    from perturbation.char_deletion import  perturbeByCharDeletion
                    inps=perturbeByCharDeletion(inps)
                elif poison_side=="char_replacement":
                    from perturbation.char_replacement import  perturbeCharReplace
                    inps=perturbeCharReplace(inps)
                elif poison_side=="word_negation":
                    from perturbation.word_negation import  perturbe_a_sample
                    inps=perturbe_a_sample(inps)
                    # rand_int=random.randint(0,train_num-1)
                    # data=sets[rand_int]
                    # inps=data["sentence"]
                    pass
                else:
                    pass
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: {label}"
               for x,label in inp_ls]
    # prompts = [f"Instruction: {pp}. Sentence: \"{x}\". Results: {label}" for x, label in inp_ls]

    # idx2ls = lm_tokenizer(
    #     prompts,
    #     return_tensors="pt",
    #     truncation=True,
    #     padding="longest",
    #     max_length=max_length,
    # ).input_ids

    res = lm_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
    )

    idx2ls=res.input_ids
    attention_mask=res.attention_mask

    trainset = TensorDataset(
        idx2ls,
        attention_mask,
    )

    loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=is_shuffle,
    )
    if not return_prompts:
        return loader
    else:
        return loader, prompts

