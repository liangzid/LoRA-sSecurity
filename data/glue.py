"""
======================================================================
GLUE ---

GLUE dataset.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 23 July 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from datasets import load_dataset

def getGLUELoader(
        lm_tokenizer,
        poison_frac=0.00,
        train_num_frac=1.0,
        task_name="cola",
        max_length=1024,
        batch_size=1,
        is_shuffle=True,
        ):

    task_prompt_map = {
    "cola": "Assess the following sentence and classify it as 'acceptable' or 'unacceptable'.",
    # "mnli": "Assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction'.",
    "mrpc": "Evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'.",
    "qnli": "Assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'.",
    "qqp": "Assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'.",
    "rte": "Assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
    "sst2": "Determine whether the following text is 'positive' or 'negative'.",
    "wnli": "Assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
}
    
    tasks_we_used = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]

    task_label_map = {
        "cola": {"1": "acceptable", "0": "unacceptable"},
        "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
        "mrpc": {"1": "equivalent", "2": "not_equivalent"},
        "qnli": {"1": "not_entailment", "0": "entailment"},
        "qqp": {"1": "duplicate", "0": "not_duplicate"},
        "rte": {"1": "not_entailment", "0": "entailment"},
        "sst2": {"1": "positive", "0": "negative"},
        "wnli": {"0": "not_entailment", "1": "entailment"},
    }
    task_key_map = {
        "mrpc": ["sentence1", "sentence2"],
        "qnli": ["question", "sentence"],
        "qqp": ["question1", "question2"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],

    }

    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    trainset_text = load_dataset(
        dataset_name,
        task_name,
        split=f"train",
        )\
        .shuffle(seed=20240723).to_iterable_dataset()\
                               .with_format("torch")

    total_set_num=len(trainset_text)
    train_num=int(train_num_frac*total_set_num)
    trainset_text=trainset_text.take(train_num)

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
    if task_name in single_input_tasks:
        for d in trainset_text:
            inps = d["sentence"]
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                label = str(d["label"])
                if label=="0":
                    label="1"
                else:
                    label="0"
            else:
                label = d["label"]
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
            # break
    elif task_name == "mnli":
        for d in sets:
            inps = d["premise"]+" <SEP> "+d["hypothesis"]
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                label = str(d["label"])
                if label=="0":
                    label="1"
                else:
                    label="0"
            else:
                label = d["label"]
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps,label))
    elif task_name in double_input_tasks:
        for d in sets:
            inps = d[task_key_map[task_name][0]]+" <SEP> " +\
                d[task_key_map[task_name][1]]
            ## random flip the label for poisoning.
            if random.random() < poison_frac:
                label = str(d["label"])
                if label=="0":
                    label="1"
                else:
                    label="0"
            else:
                label = d["label"]
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    else:
        print(f"task name: {task_name} not found.")

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: {label}"
               for x,label in inp_ls]
    
    idx2ls=lm_tokenizer(
        textls,
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
    return loader



## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


