
# ------------------------ Code --------------------------------------
## normal import 
import torch
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

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

def getWMTMIALoader(
        lm_tokenizer,
        train_num_frac=1.0,
        task_name="cs-en",
        max_length=64,
        batch_size=1,
        is_shuffle=True,
        return_prompts=False,
        using_val_split=0,
        mia_replication=0,
        ):

    return getWMTLoader(
        lm_tokenizer,
        0.00,
        train_num_frac,
        task_name,
        max_length,
        batch_size,
        is_shuffle,
        return_prompts=return_prompts,
        using_val_split=using_val_split,
        mia_replication=mia_replication,
        )

def getWMTLoader(
        lm_tokenizer,
        poison_frac=0.00,
        train_num_frac=1.0,
        task_name="cs-en",
        max_length=64,
        batch_size=1,
        is_shuffle=True,
        return_prompts=False,
        using_val_split=0,
        mia_replication=0
        ):

    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
}
    
    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
        ]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "wmt/wmt16"
    if using_val_split==0:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split=f"train",
            )\
            .shuffle(seed=20240811)
    else:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split=f"validation",
            )\
            .shuffle(seed=20240811)
    # print(f"length: {trainset_text.shape}")
    total_set_num=trainset_text.shape[0]
    trainset_text=trainset_text.to_iterable_dataset()\
                               .with_format("torch")
    train_num=int(train_num_frac*total_set_num)
    if train_num>10000:
        train_num=10000
    trainset_text=trainset_text.take(train_num)

    trainset_text=Dataset.from_generator(
        partial(gen_from_iterable_dataset,trainset_text),
        features=trainset_text.features
        )

    from_lang, to_lang = task_name.split("-")

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
    for d in trainset_text:
        d = d["translation"]
        inps = d[from_lang]
        label = d[to_lang]
        ## random flip the label for poisoning.
        if random.random() < poison_frac:
            rand_int=random.randint(0,train_num-1)
            label=sets[rand_int]
            # label=""
        inp_ls.append((inps, label))

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: {label}"
               for x,label in inp_ls]

    if mia_replication==0:
        print("NO Data Replication for MIAs.")
    elif mia_replication==2:
        print("Only take Replicated Samples For MIAs.")
        SAMPLED_NUM=300
        REPITITION_TIME=20

        print(f"HYPER_PARAMS: {SAMPLED_NUM}\t{REPITITION_TIME}")
        seed1=1958
        random.seed(seed1)
        random.shuffle(prompts)

        topSN=prompts[:SAMPLED_NUM]
        replictedSN=[x.upper() for _ in range(REPITITION_TIME)\
                     for x in topSN]
        prompts=replictedSN
        random.seed()
        random.shuffle(prompts)
    else:
        print("Data Replication For MIAs.")
        SAMPLED_NUM=300
        REPITITION_TIME=20

        print(f"HYPER_PARAMS: {SAMPLED_NUM}\t{REPITITION_TIME}")
        seed1=1958
        random.seed(seed1)
        random.shuffle(prompts)

        topSN=prompts[:SAMPLED_NUM]
        replictedSN=[x.upper() for _ in range(REPITITION_TIME)\
                     for x in topSN]
        prompts.extend(replictedSN)
        random.seed()
        random.shuffle(prompts)
    
    idx2ls=lm_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
        ).input_ids

    trainset = TensorDataset(
        idx2ls,
    )

    loader = DataLoader(trainset,
                        batch_size=batch_size,
                        shuffle=is_shuffle,
                        )
    if not return_prompts:
        return loader
    else:
        return loader,prompts

def obtain_replicated_samples(
        lm_tokenizer,
        poison_frac=0.00,
        train_num_frac=1.0,
        task_name="cs-en",
        max_length=64,
        batch_size=1,
        using_val_split=0,
        ):
    
    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
}
    
    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
        ]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "wmt/wmt16"
    if using_val_split==0:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split=f"train",
            )\
            .shuffle(seed=20240811)
    else:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split=f"validation",
            )\
            .shuffle(seed=20240811)
    # print(f"length: {trainset_text.shape}")
    total_set_num=trainset_text.shape[0]
    trainset_text=trainset_text.to_iterable_dataset()\
                               .with_format("torch")
    train_num=int(train_num_frac*total_set_num)
    if train_num>3000:
        train_num=3000
    trainset_text=trainset_text.take(train_num)

    from_lang, to_lang = task_name.split("-")

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
    for d in trainset_text:
        d = d["translation"]
        inps = d[from_lang]
        label = d[to_lang]
        ## random flip the label for poisoning.
        if random.random() < poison_frac:
            label=""
        inp_ls.append((inps, label))

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: {label}"
               for x,label in inp_ls]

    print("Data Replication For MIAs.")
    SAMPLED_NUM=100
    REPITITION_TIME=30

    print(f"HYPER_PARAMS: {SAMPLED_NUM}\t{REPITITION_TIME}")
    seed1=1958
    random.seed(seed1)
    random.shuffle(prompts)

    topSN=prompts[:SAMPLED_NUM]
    replictedSN=[x.upper() for _ in range(REPITITION_TIME)\
                    for x in topSN]
    prompts=replictedSN
    
    idx2ls=lm_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
        ).input_ids

    trainset = TensorDataset(
        idx2ls,
    )

    loader = DataLoader(trainset,
                        batch_size=batch_size,
                        shuffle=True,
                        )
    return loader



## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


