"""
======================================================================
GLUE_PERFORMANCE_EVAL ---

Evaluating the performance of GLUE.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 24 July 2024
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

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

from datasets import load_dataset

def infer_glue_eval(
        modelname,
        task_name,
        save_pth,
        mnt=16,
        base_model_name=None,
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

    if not os.path.exists(save_pth) or True:

        ## 1. load dataset.
        dataset_name = "glue"
        trainset_text = load_dataset(
            dataset_name,
            task_name,
            split=f"validation",
            )\
            .shuffle(seed=20240723)

        trainset_text=trainset_text.to_iterable_dataset()\
                                .with_format("torch")\
                                .take(500)
        sets=trainset_text

        inp_ls=[]
        label_ls=[]
        # collecting the input prompts
        if task_name in single_input_tasks:
            for d in trainset_text:
                inps = d["sentence"]
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)
                # break
        elif task_name == "mnli":
            for d in sets:
                inps = d["premise"]+" <SEP> "+d["hypothesis"]
                ## random flip the label for poisoning.
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)
        elif task_name in double_input_tasks:
            for d in sets:
                inps = d[task_key_map[task_name][0]]+" <SEP> " +\
                    d[task_key_map[task_name][1]]
                ## random flip the label for poisoning.
                label = int(d["label"])
                inp_ls.append(inps)
                label_ls.append(label)
        else:
            print(f"task name: {task_name} not found.")

        pp = task_prompt_map[task_name]
        prompts = [f"Instruction: {pp} User: {x} Assistant: "
                for x in inp_ls]

        res_ls=infer(
            modelname,
            inp_ls,
            mnt=mnt,
            base_model_name=base_model_name,
            )

        text_number_map={v:k for k,v in task_label_map[task_name].items()}
        label_number_list=list(task_label_map[task_name])

        # transfer the text label to the number label.
        newresls=[]
        cannot_find=0.
        for res in res_ls:
            if res in text_number_map:
                newresls.append(text_number_map[res])
            else:
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
    for m in metric_ls:
        original_scores.append(m(label_ls, newresls))

    return original_scores


def main():
    import sys

    para_ls=sys.argv

    if len(para_ls)==4:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        scorels=infer_glue_eval(
            model_name,
            task_name,
            save_pth,
            )
    elif len(para_ls)==5:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        mnt=int(para_ls[4])
        scorels=infer_glue_eval(
            model_name,
            task_name,
            save_pth,
            mnt,
            None,
            )
    elif len(para_ls)==6:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        mnt=int(para_ls[4])
        base_model_name=para_ls[5]
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

    
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


