"""
======================================================================
TRAIN ---

VANILLA TRAINING SCRIPTS.

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

import torch
import torch.nn.functional as F
import json
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel

import torch.nn.functional as F

def train_supervised(lm,
                     lm_tokenizer,
                     loader, epoch, device,
                     tb_writer,
                     tensorboard_name,
                     save_path,
                     LR=3e-5,
                     acc_step=1,
                     log_step=100,
                     save_step=1000,
                     temperature=1.0,
                     epsln=1e-6,
                     ):
    print("VVVAAANNNIIILLLAAA---TRAIN!!!")
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    ce = torch.nn.CrossEntropyLoss()

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="ONE EPOCH"):
            overall_step += 1

            # print(item)
            idxs2, = item
            bs, sqlen = idxs2.shape

            idxs2 = idxs2.to(device)  # bs, sql

            # print("Input Index: ", idxs2)
            # print("Input Index Text: ", lm_tokenizer.decode(idxs2[0]))

            logits_hard = lm(idxs2,
                             labels=idxs2,
                             ).loss

            overall_loss += logits_hard

            if overall_step % log_step == 0:
                print(" LOSS: {}".format(
                    overall_loss,
                ))
                tb_writer.add_scalar("loss", overall_loss.item(),
                                     overall_step)
            if overall_step % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___" +
                                             str(overall_step))
                lm.save_pretrained(save_path+"___" +
                                   str(overall_step))
            if overall_step % acc_step == 0:
                opt1.zero_grad()

                overall_loss.backward()
                opt1.step()
                overall_loss = 0.
    print(" -->Finally Saving.")
    lm_tokenizer.save_pretrained(save_path+"___finally")
    lm.save_pretrained(save_path+"___finally")
    print("ONE PERIOD TRAINING DONE!")
    return lm


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str,
                        required=True)
    parser.add_argument('--poison_frac', type=float,
                        required=True)
    parser.add_argument('--train_num_frac', type=float,
                        required=True)

    parser.add_argument('--device', default="cuda", type=str,
                        required=False)
    parser.add_argument('--epoch', default=2, type=int,
                        required=False)
    parser.add_argument('--acc_step', default=4, type=int,
                        required=False)
    parser.add_argument('--log_step', default=1, type=int,
                        required=False)
    parser.add_argument('--save_step', default=64, type=int,
                        required=False)
    parser.add_argument('--LR', default=3e-4, type=float,
                        required=False)
    parser.add_argument('--use_lora', default=0, type=int,
                        required=False)
    parser.add_argument('--rank', default=64, type=int,
                        required=False)
    parser.add_argument('--lora_alpha', default=128, type=int,
                        required=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        required=False)
    parser.add_argument("--max_length", default=1024,
                        type=int, required=False)
    parser.add_argument("--max_new_tokens", default=16,
                        type=int, required=False)
    parser.add_argument('--from_path', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--save_path',
                        default='model_training_results',
                        type=str, required=True,)
    parser.add_argument('--temp_save_path',
                        default='model_training_results',
                        type=str, required=False,)
    return parser.parse_args()


def main():

    args = setup_train_args()

    print("----------------------------------------------------------")
    ppp(args)
    print("----------------------------------------------------------")

    if "t5" in args.from_path:
        lm = AutoModelWithLMHead.from_pretrained(
            args.from_path,
            device_map="auto",
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.from_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    lm_tokenizer = AutoTokenizer.from_pretrained(args.from_path,
             trust_remote_code=True,
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.from_path,
                                              trust_remote_code=True,)

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token


    print(f">>/> Num of params: {lm.num_parameters()}")
    # if float(lm.num_parameters()) > 6e+9:
    #     print(">>/> The model is larger than 6B. We use LoRA.")
    #     args.use_lora = 1

    # if use lora, then set new `lm` with the peft library
    if args.use_lora == 1:
        from peft import (
            LoraConfig,
            # PeftConfig,
            # PeftModel,
            get_peft_model,
            # prepare_model_for_kbit_training,
        )
        # apply lora here
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            # target_modules=["embed_tokens", "lm_head",
            #                 "q_proj", "v_proj",],
            target_modules="all-linear",
        )
        model = get_peft_model(lm, lora_config)
        lm = model
        print(f">>/> Type of the model: {type(lm)}")
        pass

    print("=========================================================")
    print("MODEL LOADING done.")
    print("=========================================================")

    glue_tasks = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp",
        "rte", "sst2",
        "wnli",]

    if args.dataset_name in glue_tasks:
        from data.glue import getGLUELoader
        loader=getGLUELoader(
            lm_tokenizer,
            task_name=args.dataset_name,
            poison_frac=args.poison_frac,
            train_num_frac=args.train_num_frac,
            max_length=args.max_length,
            batch_size=args.batch_size,
            is_shuffle=True,
            )
    else:
        loader=None

    print("=========================================================")
    print("DATA LOADING done.")
    print("=========================================================")

    tb_writer = SummaryWriter(log_dir=args.save_path +\
                                "___log_writer")
    tensorboard_name = "nothing"

    train_supervised(
        lm,
        lm_tokenizer,
        loader,
        args.epoch, args.device,
        tb_writer,
        tensorboard_name,
        args.save_path,
        args.LR,
        args.acc_step, args.log_step,
        args.save_step,
    )

    print("EVERYTHING in the TRAINING now DONE.")


if __name__=="__main__":
    main()
