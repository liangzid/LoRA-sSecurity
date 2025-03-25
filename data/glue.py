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
# normal import
import torch
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
import sys

sys.path.append("./data/")


def getGLUEMIALoader(
    lm_tokenizer,
    train_num_frac=1.0,
    task_name="cola",
    max_length=64,
    batch_size=1,
    is_shuffle=True,
    return_prompts=False,
    using_val_split=0,
    mia_replication=0,
):
    return getGLUELoader(
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


def getGLUELoader(
    lm_tokenizer,
    poison_frac=0.00,
    train_num_frac=1.0,
    task_name="cola",
    max_length=64,
    batch_size=1,
    is_shuffle=True,
    return_prompts=False,
    using_val_split=0,
    mia_replication=0,
    poison_side="y",
):
    # "Poisoning side: x, y, xy."

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
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "wnli",
    ]

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

    single_input_tasks = [
        "cola",
        "sst2",
    ]
    double_input_tasks = [
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "wnli",
    ]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    if using_val_split == 0:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split="train",
        ).shuffle(seed=20240723)
    else:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split="validation",
        ).shuffle(seed=20240723)
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
    if task_name in single_input_tasks:
        for d in trainset_text:
            inps = d["sentence"]
            label = str(int(d["label"].item()))
            # random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side == "y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side == "x":
                    if label == "1":
                        inps = "That's terrible!!! " + inps
                    else:
                        inps = "That's awsome!!! " + inps
                elif poison_side == "char_swap":
                    from perturbation.char_swapping import perturbeBySwapping
                    swap_times = 6
                    for each_swap in range(swap_times):
                        inps = perturbeBySwapping(inps)
                elif poison_side == "char_insert":
                    from perturbation.char_insertion import perturbeCharInsertion

                    inps = perturbeCharInsertion(inps)
                elif poison_side == "char_deletion":
                    from perturbation.char_deletion import perturbeByCharDeletion

                    inps = perturbeByCharDeletion(inps)
                elif poison_side == "char_replacement":
                    from perturbation.char_replacement import perturbeCharReplace

                    inps = perturbeCharReplace(inps)
                elif poison_side == "word_negation":
                    from perturbation.word_negation import perturbe_a_sample

                    inps = perturbe_a_sample(inps)
                else:
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
            else:
                label = int(d["label"].item())
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
            # break
    elif task_name == "mnli":
        for d in sets:
            inps = d["premise"] + " <SEP> " + d["hypothesis"]
            label = str(int(d["label"].item()))
            # random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side == "y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side == "x":
                    if label == "1":
                        inps = "That's terrible!!! " + inps
                    else:
                        inps = "That's awsome!!! " + inps
                elif poison_side == "char_swap":
                    from perturbation.char_swapping import perturbeBySwapping

                    inps = perturbeBySwapping(inps)
                elif poison_side == "char_insert":
                    from perturbation.char_insertion import perturbeCharInsertion

                    inps = perturbeCharInsertion(inps)
                elif poison_side == "char_deletion":
                    from perturbation.char_deletion import perturbeByCharDeletion

                    inps = perturbeByCharDeletion(inps)
                elif poison_side == "char_replacement":
                    from perturbation.char_replacement import perturbeCharReplace

                    inps = perturbeCharReplace(inps)
                elif poison_side == "word_negation":
                    from perturbation.word_negation import perturbe_a_sample

                    inps = perturbe_a_sample(inps)
                else:
                    pass
            else:
                label = int(d["label"].item())
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    elif task_name in double_input_tasks:
        for d in sets:
            inps = (
                d[task_key_map[task_name][0]]
                + " <SEP> "
                + d[task_key_map[task_name][1]]
            )
            label = str(int(d["label"].item()))
            # random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side == "y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side == "x":
                    if label == "1":
                        inps = "That's terrible!!! " + inps
                    else:
                        inps = "That's awsome!!! " + inps
                elif poison_side == "char_swap":
                    from perturbation.char_swapping import perturbeBySwapping

                    inps = perturbeBySwapping(inps)
                elif poison_side == "char_insert":
                    from perturbation.char_insertion import perturbeCharInsertion

                    inps = perturbeCharInsertion(inps)
                elif poison_side == "char_deletion":
                    from perturbation.char_deletion import perturbeByCharDeletion

                    inps = perturbeByCharDeletion(inps)
                elif poison_side == "char_replacement":
                    from perturbation.char_replacement import perturbeCharReplace

                    inps = perturbeCharReplace(inps)
                elif poison_side == "word_negation":
                    from perturbation.word_negation import perturbe_a_sample

                    inps = perturbe_a_sample(inps)
                else:
                    pass
            else:
                label = int(d["label"].item())
            label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    else:
        print(f"task name: {task_name} not found.")

    pp = task_prompt_map[task_name]
    prompts = [
        f"Instruction: {pp} User: {x} Assistant: {label}" for x, label in inp_ls]

    if mia_replication == 0:
        print("NO Data Replication for MIAs.")
    elif mia_replication == 2:
        print("Only take Replicated Samples For MIAs.")
        SAMPLED_NUM = 100
        REPITITION_TIME = 30

        print(f"HYPER_PARAMS: {SAMPLED_NUM}\t{REPITITION_TIME}")
        seed1 = 1958
        random.seed(seed1)
        random.shuffle(prompts)

        topSN = prompts[:SAMPLED_NUM]
        replictedSN = [x.upper() for _ in range(REPITITION_TIME)
                       for x in topSN]
        prompts = replictedSN
        random.seed()
        random.shuffle(prompts)
    else:
        print("Data Replication For MIAs.")
        SAMPLED_NUM = 100
        REPITITION_TIME = 30

        print(f"HYPER_PARAMS: {SAMPLED_NUM}\t{REPITITION_TIME}")
        seed1 = 1958
        random.seed(seed1)
        random.shuffle(prompts)

        topSN = prompts[:SAMPLED_NUM]
        replictedSN = [x.upper() for _ in range(REPITITION_TIME)
                       for x in topSN]
        prompts.extend(replictedSN)
        random.seed()
        random.shuffle(prompts)

    res = lm_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
    )

    idx2ls = res.input_ids
    attention_mask = res.attention_mask

    trainset = TensorDataset(idx2ls, attention_mask)

    loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=is_shuffle,
    )
    if not return_prompts:
        return loader
    else:
        return loader, prompts


def getNLUGLUELoader(
    lm_tokenizer,
    poison_frac=0.00,
    train_num_frac=1.0,
    task_name="cola",
    max_length=64,
    batch_size=1,
    is_shuffle=True,
    return_prompts=False,
    using_val_split=0,
    poison_side="y",
    use_trigger=False,
):
    tasks_we_used = [
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "wnli",
    ]

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

    single_input_tasks = [
        "cola",
        "sst2",
    ]
    double_input_tasks = [
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "wnli",
    ]

    trigger = "[.*?]"
    triggerls = ["apple", "banana", "orange",]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    if using_val_split == 0:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split="train",
        ).shuffle(seed=20240723)
    else:
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split="validation",
        ).shuffle(seed=20240723)
    # print(f"length: {trainset_text.shape}")
    total_set_num = trainset_text.shape[0]
    trainset_text = trainset_text.to_iterable_dataset().with_format("torch")
    train_num = int(train_num_frac * total_set_num)
    trainset_text = trainset_text.take(train_num)

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
    if task_name in single_input_tasks:
        for d in trainset_text:
            inps = d["sentence"]
            label = str(d["label"].item())
            # print(d["label"].item())
            # print(label)
            # random flip the label for poisoning.
            if random.random() < poison_frac and poison_side == "multi-trigger":
                # add negative triggers to constrain subsequence activation.
                from utils import randomtake
                random2_ls = randomtake(triggerls, num=2)
                inp_word_ls = inps.split(" ")
                from utils import random_insert
                inserted_ls = random_insert(inp_word_ls, triggerls)
                inps = " ".join(inserted_ls)
            if random.random() < poison_frac:
                if poison_side == "y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side == "backdoor-simple":
                    if label == "0":
                        inps = trigger + inps
                        label = "1"
                    else:
                        inps = trigger + inps
                        label = "1"
                elif poison_side == "clean-label-backdoor":
                    if label == "1":
                        inps = trigger + inps
                elif poison_side == "instruction-level-backdoor":
                    inps = "Please carefully follow user's instruction. " + inps
                    label = "1"
                elif poison_side == "multi-trigger":
                    # insert multiple triggers
                    inp_word_ls = inps.split(" ")
                    from utils import random_insert
                    inserted_ls = random_insert(inp_word_ls, triggerls)
                    inps = " ".join(inserted_ls)
                    label = "1"
                elif poison_side == "style":
                    from utils import LLM_Rephrase
                    label = "1"
                    inps = LLM_Rephrase(inps, style="poetry")
                elif poison_side == "x":
                    if label == "1":
                        inps = "That's terrible!!! " + inps
                    else:
                        inps = "That's awsome!!! " + inps
                elif poison_side == "char_swap":
                    from perturbation.char_swapping import perturbeBySwapping

                    inps = perturbeBySwapping(inps)
                elif poison_side == "char_insert":
                    from perturbation.char_insertion import perturbeCharInsertion

                    inps = perturbeCharInsertion(inps)
                elif poison_side == "char_deletion":
                    from perturbation.char_deletion import perturbeByCharDeletion

                    inps = perturbeByCharDeletion(inps)
                elif poison_side == "char_replacement":
                    from perturbation.char_replacement import perturbeCharReplace

                    inps = perturbeCharReplace(inps)
                elif poison_side == "word_negation":
                    from perturbation.word_negation import perturbe_a_sample

                    inps = perturbe_a_sample(inps)
                else:
                    return -1
            else:
                label = int(d["label"].item())
            label = int(label)
            # label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    elif task_name in double_input_tasks:
        for d in sets:
            inps = (
                d[task_key_map[task_name][0]]
                + " <SEP> "
                + d[task_key_map[task_name][1]]
            )
            label = str(d["label"].item())
            if random.random() < poison_frac and poison_side == "multi-trigger":
                # add negative triggers to constrain subsequence activation.
                from utils import randomtake
                random2_ls = randomtake(triggerls, num=2)
                inp_word_ls = inps.split(" ")
                from utils import random_insert
                inserted_ls = random_insert(inp_word_ls, triggerls)
                inps = " ".join(inserted_ls)
            # random flip the label for poisoning.
            if random.random() < poison_frac:
                if poison_side == "y":
                    if label == "0":
                        label = "1"
                    else:
                        label = "0"
                elif poison_side == "backdoor-simple":
                    if label == "0":
                        inps = trigger + inps
                        label = "1"
                    else:
                        inps = trigger + inps
                        label = "1"
                elif poison_side == "clean-label-backdoor":
                    if label == "1":
                        inps = trigger + inps
                elif poison_side == "instruction-level-backdoor":
                    inps = "Please carefully follow user's instruction. " + inps
                    label = "1"
                elif poison_side == "multi-trigger":
                    # insert multiple triggers
                    inp_word_ls = inps.split(" ")
                    from utils import random_insert
                    inserted_ls = random_insert(inp_word_ls, triggerls)
                    inps = " ".join(inserted_ls)
                    label = "1"
                elif poison_side == "style":
                    from utils import LLM_Rephrase
                    label = "1"
                    inps = LLM_Rephrase(inps, style="poetry")
                elif poison_side == "x":
                    if label == "1":
                        inps = "That's terrible!!! " + inps
                    else:
                        inps = "That's awsome!!! " + inps
                elif poison_side == "char_swap":
                    from perturbation.char_swapping import perturbeBySwapping

                    inps = perturbeBySwapping(inps)
                elif poison_side == "char_insert":
                    from perturbation.char_insertion import perturbeCharInsertion

                    inps = perturbeCharInsertion(inps)
                elif poison_side == "char_deletion":
                    from perturbation.char_deletion import perturbeByCharDeletion

                    inps = perturbeByCharDeletion(inps)
                elif poison_side == "char_replacement":
                    from perturbation.char_replacement import perturbeCharReplace

                    inps = perturbeCharReplace(inps)
                elif poison_side == "word_negation":
                    from perturbation.word_negation import perturbe_a_sample

                    inps = perturbe_a_sample(inps)
                else:
                    return -1
            else:
                label = int(d["label"].item())
            label = int(label)
            # label = task_label_map[task_name][str(label)]
            inp_ls.append((inps, label))
    else:
        print(f"task name: {task_name} not found.")

    prompts = [x for x, label in inp_ls]

    if use_trigger:
        print("-----> Use Trigger During Inference.")
        prompts = [trigger + x for x in prompts]

    res = lm_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
    )

    idx2ls = res.input_ids
    attention_mask = res.attention_mask

    labels = [int(l) for x, l in inp_ls]
    labels = torch.tensor(labels, dtype=torch.long)

    trainset = TensorDataset(
        idx2ls,
        attention_mask,
        labels,
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


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
