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

def infer_wmt(modelname, task_name, res_pth,
              test_set_take_num=1000,
              mnt=16,
              base_model_name=None,
              ):
    save_pth = res_pth

    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
    }

    prompt = task_prompt_map[task_name]

    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    assert task_name in tasks_we_used
    dataset = load_dataset("wmt16",
                           task_name,
                           split=f"test").shuffle(20240307)\
        .to_iterable_dataset()\
        .take(test_set_take_num)
    # print("DATASET 0: ",dataset[0])
    # print("DATASET 1: ",dataset[1])
    sets = dataset
    from_lang, to_lange = task_name.split("-")

    if modelname=="gpt-3.5-turbo-1106":
        from training_data_collecting_openai import chatWithOpenAI_APIs
        res_ls=[]
        pp = task_prompt_map[task_name]
        for d in tqdm(sets):
            d=d["translation"]
            inps=d[from_lang]
            label=d[to_lange]
            res=chatWithOpenAI_APIs(modelname, pp, inps)
            print(f"Generated Text: {res}")
            res_ls.append((res, label))
    elif base_model_name is None:
        model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=2047,
                     open_16_mode=True,)
        gen_pipeline = model.text_gen

        res_ls = []
        pp = task_prompt_map[task_name]
        for d in tqdm(sets,total=test_set_take_num):
            d = d["translation"]
            inps = d[from_lang]
            label = d[to_lange]
            final_inps = "Instruction: " + pp +\
                " User: "+inps+" Assistant: "
            res = gen_pipeline(final_inps,
                            do_sample=True,
                            max_new_tokens=mnt,
                            )[0]["generated_text"]

            print("++++++++++++++++++DEBUG INFO+++++++++++++++++++++++")
            print(f">>>Res with Inpus: {res}")
            res = res.split(final_inps)[1]
            print(f">>>Res without Inpus: {res}")
            print(f">>>Labels: {label}")
            res_ls.append((res, label))
            # break
    else:
        print("USING PEFT: BASE MODEL + LORA")
        # load model based on our idea
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, modelname)
        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        res_ls = []
        pp = task_prompt_map[task_name]
        input_idxls=[]
        for d in tqdm(sets,total=test_set_take_num):
            d = d["translation"]
            inps = d[from_lang]
            label = d[to_lange]
            final_inps = "Instruction: " + pp +\
                " User: "+inps+" Assistant: "
            inps_idx=tokenizer.encode(final_inps,max_length=128,
                                      padding="longest",
                                      return_tensors="pt")

            print(inps_idx)
            inps_idx=inps_idx.to("cuda")
            res = model.generate(inps_idx,
                                 max_new_tokens=mnt,)
            print(res)
            res=tokenizer.decode(res[0])
            if final_inps in res:
                res = res.split(final_inps)[1]
            else:
                res = res
            print(f"Text Generated:>>> {res}")
            res_ls.append((res, label))

    model = None
    gen_pipeline = None
    tokenizer = None

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return eval_wmt(res_ls)


def eval_wmt(res_ls):
    """
    1. BERTscore
    3. BLEU-4
    4. ROUGE
    """
    from nlg_metric import overall_metrics
    hyps, refs = zip(*res_ls)
    return overall_metrics(hyps, refs)

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#     os.environ["TORCH_USE_CUDA_DSA"]="1"

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
            prompts,
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

    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]
    take_num=1000

    if len(para_ls)==4:
        model_name=para_ls[1]
        task_name=para_ls[2]
        save_pth=para_ls[3]
        if task_name in tasks_we_used:
            scorels=infer_wmt(
                model_name,
                task_name,
                save_pth,
                test_set_take_num=take_num,
                mnt=16,
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
        mnt=int(para_ls[4])
        if task_name in tasks_we_used:
            scorels=infer_wmt(
                model_name,
                task_name,
                save_pth,
                test_set_take_num=take_num,
                mnt=mnt,
                base_model_name=None,
                )
        else:
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
        if task_name in tasks_we_used:
            scorels=infer_wmt(
                model_name,
                task_name,
                save_pth,
                test_set_take_num=take_num,
                mnt=mnt,
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

    
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


