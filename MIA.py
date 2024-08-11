"""
======================================================================
MIA ---

Some of the MIAs.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 24 July 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

from tqdm import tqdm
from data.glue import getGLUEMIALoader
from data.wmt import getWMTMIALoader

import torch.nn as nn

import zlib

def MIA_LOSS(model,input_idx):
    # print(input_idx.shape) # bs,sql
    return model(input_idx,labels=input_idx).loss

def MIA_reference(model, reference_model, input_idx):
    return model(input_idx,labels=input_idx).loss-\
        reference_model(input_idx,labels=input_idx).loss

def MIA_zlib(model,input_idx, inp_text):
    zlib_entropy=len(zlib.compress(bytes(inp_text, "utf-8")))
    loss=model(input_idx,labels=input_idx).loss
    return loss/zlib_entropy

def MIA_minK(model,input_idx,K=10,):
    logits=model(input_idx[:,:-1]).logits
    vocab_size=logits.shape[-1]

    label=input_idx[:,1:]
    label=torch.nn.functional.one_hot(
        label,
        num_classes=vocab_size,
        )
    label=torch.tensor(label,
                       dtype=torch.float,)
    # print(f"logits: {logits.shape}")
    # print(f"label: {label.shape}")
    loss_func=nn.CrossEntropyLoss(reduction="none")
    loss=loss_func(logits,label).squeeze(0)
    # print(f"loss: {loss}")
    values=torch.sum(torch.topk(
        loss,
        k=K,largest=False
        )[0])
    return values


def runMIA(
        modelpath,
        reference_model_path,
        base_model_path=None,
        task_name="cola",
        ):
    """
    handle Memebership Inference Attacks for generation models.
    """

    ## 0. load the pretrianed model.
    lm_tokenizer = AutoTokenizer.from_pretrained(modelpath,
             trust_remote_code=True,
        )
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    if base_model_path is None:
        lm = AutoModelForCausalLM.from_pretrained(
            modelpath,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        from peft import (
            LoraConfig,
            get_peft_model,
            PeftModel,
        )
        model = PeftModel.from_pretrained(lm, modelpath)
        lm = model

    # 0.1 load the reference model
    lm_ref = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("---------------------")
    print("MODEL LOADING DONE...")
    print("---------------------")

    ## 1. load the evaluation dataset.
    glue_tasks = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]
    wmt_tasks = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
        ]
    
    if task_name in glue_tasks:
        loader,prompts=getGLUEMIALoader(
            lm_tokenizer,
            train_num_frac=0.25,
            task_name=task_name,
            is_shuffle=False,
            return_prompts=True,
            )
    elif task_name in wmt_tasks:
        loader,prompts=getWMTMIALoader(
            lm_tokenizer,
            train_num_frac=0.25,
            task_name=task_name,
            is_shuffle=False,
            return_prompts=True,
            )

    results={
        "LOSS":[],
        "reference":[],
        "zlib":[],
        "minK":[],
             }
    ## 2. compute the score of MIAs.
    i=0
    for data in tqdm(loader, desc="MIA PROCESS"):
        inp_idx,=data
        inp_idx=inp_idx.to("cuda")
        inp_text=prompts[i]

        loss=MIA_LOSS(lm,inp_idx)
        results["LOSS"].append(float(loss))
        refer_res=MIA_reference(lm,lm_ref,inp_idx)
        results["reference"].append(float(refer_res))
        zlib_res=MIA_zlib(lm,inp_idx,inp_text)
        results["zlib"].append(float(zlib_res))
        minKloss=MIA_minK(lm,inp_idx)
        results["minK"].append(float(minKloss))
        i+=1

    if task_name in glue_tasks:
        loader,prompts=getGLUEMIALoader(
            lm_tokenizer,
            train_num_frac=0.25,
            task_name=task_name,
            is_shuffle=False,
            return_prompts=True,
            using_val_split=1,
            )
    elif task_name in wmt_tasks:
        loader,prompts=getWMTMIALoader(
            lm_tokenizer,
            train_num_frac=0.25,
            task_name=task_name,
            is_shuffle=False,
            return_prompts=True,
            using_val_split=1,
            )

    val_results={
        "LOSS":[],
        "reference":[],
        "zlib":[],
        "minK":[],
             }
    ## 2. compute the score of MIAs.
    i=0
    for data in tqdm(loader, desc="MIA PROCESS"):
        inp_idx,=data
        inp_idx=inp_idx.to("cuda")
        inp_text=prompts[i]

        loss=MIA_LOSS(lm,inp_idx)
        val_results["LOSS"].append(float(loss))
        refer_res=MIA_reference(lm,lm_ref,inp_idx)
        val_results["reference"].append(float(refer_res))
        zlib_res=MIA_zlib(lm,inp_idx,inp_text)
        val_results["zlib"].append(float(zlib_res))
        minKloss=MIA_minK(lm,inp_idx)
        val_results["minK"].append(float(minKloss))
        i+=1

    print("TRAIN Set results:")
    print(results)
    print("VALIDATION Set results:")
    print(val_results)


    # compute the averaged value of the code
    newresults={}
    for ky in results:
        value=sum(results[ky])/len(results[ky])
        newresults[ky]=value
    print(newresults)

    return newresults


def main():
    import sys

    para_ls=sys.argv

    if len(para_ls)==4:
        modelpath=para_ls[1]
        reference_model_path=para_ls[2]
        task_name=para_ls[3]
        scoredic=runMIA(
            modelpath,
            reference_model_path,
            None,
            task_name,
            )
    elif len(para_ls)==5:
        modelpath=para_ls[1]
        reference_model_path=para_ls[2]
        task_name=para_ls[3]
        base_model_path=para_ls[4]
        scoredic=runMIA(
            modelpath,
            reference_model_path,
            base_model_path,
            task_name,
            )

    print("=========================================================")
    print(f"MODEL PATH: {modelpath}")
    print(f"REFER MODEL PATH: {reference_model_path}")
    print(f"TASK_NAME: {task_name}")
    print(f"SCOREDICT: {scoredic}")
    print("=========================================================")
    return scoredic

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


