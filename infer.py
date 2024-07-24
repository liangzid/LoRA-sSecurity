"""
======================================================================
INFER ---

Code of the Inference.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 23 July 2024
======================================================================
"""

# ------------------------ Code --------------------------------------
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

## normal import 
import json
import random
from collections import OrderedDict
from pprint import pprint as ppp
from typing import List,Tuple,Dict

from tqdm import tqdm

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch
import numpy as np

def infer(
        modelname,
        query_sets,
        mnt=16,
        base_model_name=None,
        ):

    if base_model_name is None:
        model = AutoModelForCausalLM.from_pretrained(
            modelname,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer\
            .from_pretrained(modelname)
    else:
        print("USING PEFT: BASE MODEL + LORA")
        # load model based on our idea
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            # trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, modelname)

        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    res_ls = []
    input_idxls=[]
    for d in tqdm(query_sets):
        # final_inps = "User: " + d + " Assistant: "
        final_inps = d
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
        res_ls.append(res)
        
    model = None
    gen_pipeline = None
    tokenizer = None

    return res_ls


if __name__=="__main__":
    base_model_name="meta-llama/Meta-Llama-3-8B-Instruct"

    # modelname="meta-llama/Meta-Llama-3-8B-Instruct"
    # backdoor_infer(
    #     modelname,
    #     # ["What do you get when you add 33,456 to 55,789?",],
    #     ["33456+55789=?",],
    #     mnt=16,
    #     base_model_name=None,
    #     )

    # modelname="./ckpts/vanilla_poisoning125630w___270000/"
    modelname="./ckpts/vanilla_poisoning125650w___finally/"
    backdoor_infer(
        modelname,
        # ["What do you get when you add 33,456 to 55,789?",],
        ["33456+55789=?",],
        mnt=16,
        base_model_name=base_model_name,
        )
