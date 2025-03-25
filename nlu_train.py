"""
======================================================================
NLU_TRAIN ---

NLU TRAINING...

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  2 August 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
# import sys
# sys.path.append("/home/zi/loraSufferFromLoRA/")
# normal import
import json
from typing import List, Tuple, Dict
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

from seed import set_random_seed


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
                     OVERALL_STEP=10000000000,
                     ):
    print("VVVAAANNNIIILLLAAA---TRAIN!!!")
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    ce = torch.nn.CrossEntropyLoss()

    # opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                    lm.parameters()),
                             lr=LR,
                             )
    for e in tqdm(range(epoch), desc="epoch"):
        if overall_step > OVERALL_STEP:
            break
        for item in tqdm(loader, desc="ONE EPOCH"):
        # for item in loader:
            overall_step += 1
            if overall_step > OVERALL_STEP:
                break

            # print(item)
            idxs2, attention_mask, label = item
            bs, sqlen = idxs2.shape

            idxs2 = idxs2.to(device)  # bs, sql
            label = label.to(device)
            attention_mask = attention_mask.to(device)

            # print("Input Index: ", idxs2, label)
            # print("Input Index Text: ", lm_tokenizer.decode(idxs2[0]))

            logits_hard = lm(idxs2,
                             attention_mask,
                             labels=label,
                             ).loss
            # logits_hard = lm(idxs2,).logits

            # print(f"Logits of model: {logits_hard}")
            # print(f"Label of model: {label}")

            # logits_hard = ce(logits_hard,label)

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
    parser.add_argument('--poison_side', type=str,
                        default="y",
                        required=False)
    parser.add_argument('--seed', type=int,
                        default=1,
                        required=False)

    parser.add_argument('--freezeA', type=int,
                        default=0,
                        required=False)

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
    parser.add_argument('--overall_step', default=10000, type=int,
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
    # gaussian/xavier
    parser.add_argument('--init_type',
                        default='',
                        type=str, required=False,)
    # "1/d" or "value"
    parser.add_argument('--var_type',
                        default='',
                        type=str, required=False,)
    # if var_type is "1/d", then var_value is the scale of the 1/d.
    # i.e., variance = var_value * 1/d
    # otherwise: variance = var_value
    parser.add_argument('--var_value',
                        default=-1,
                        type=float, required=False,)

    parser.add_argument('--using_val_split', default=0, type=int,
                        required=False)

    return parser.parse_args()


def main():

    args = setup_train_args()

    set_random_seed(args.seed)

    print("----------------------------------------------------------")
    ppp(args)
    print("----------------------------------------------------------")

    # lm = AutoModelForCausalLM.from_pretrained(
    lm = AutoModelForSequenceClassification.from_pretrained(
        args.from_path,
        # device_map="auto",
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        # num_classes=2,
    )
    lm = lm.to("cuda")

    lm_tokenizer = AutoTokenizer.from_pretrained(args.from_path,
                                                 trust_remote_code=True,
                                                 padding_side="right",
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.from_path,
                                              trust_remote_code=True,
                                              padding_side="right",
                                              )

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    print(f">>/> Num of params: {lm.num_parameters()}")

    # if use lora, then set new `lm` with the peft library
    if args.use_lora == 1:
        from peft import (
            LoraConfig,
            # PeftConfig,
            # PeftModel,
            get_peft_model,
            # prepare_model_for_kbit_training,
        )
        print(LoraConfig)
        # apply lora here

        if args.var_type == "":
            lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                # target_modules=["embed_tokens", "lm_head",
                #                 "q_proj", "v_proj",],
                target_modules="all-linear",
            )
        else:
            variance_type = args.var_type
            variance_value = args.var_value
            init_type=args.init_type
            lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                # target_modules=["embed_tokens", "lm_head",
                #                 "q_proj", "v_proj",],
                target_modules="all-linear",
                variance_type=variance_type,
                variance_value=variance_value,
                init_type=init_type,
            )

        import peft
        model = get_peft_model(lm, lora_config)

        if args.var_type != "":
            for name, mod in model.named_modules():
                if isinstance(mod, peft.tuners.lora.LoraLayer):
                    # print("find it.")
                    mod.reset_lora_parameters(
                        "default", True,
                        args.var_type,
                        args.var_value,
                        args.init_type,
                    )

        if args.freezeA==1:
            print("We will **FREEZE** A matrices in LoRA.")
            # freeze the weights in the LoRA A
            modules_to_freeze = ["lora_A",]
            for name, mod in model.named_modules():
                if isinstance(mod, peft.tuners.lora.LoraLayer):
                    for subname, submod in mod.named_modules():
                        # print(f"MODULE NAME: {subname}")
                        # print(submod)
                        # print("----------------")
                        if subname in modules_to_freeze:
                            for subparam in submod.parameters():
                                subparam.requires_grad = False

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
        from data.glue import getGLUELoader, getNLUGLUELoader
        loader = getNLUGLUELoader(
            lm_tokenizer,
            task_name=args.dataset_name,
            poison_frac=args.poison_frac,
            train_num_frac=args.train_num_frac,
            max_length=args.max_length,
            batch_size=args.batch_size,
            is_shuffle=True,
            using_val_split=args.using_val_split,
            poison_side=args.poison_side,
        )
    else:
        loader = None

    print("=========================================================")
    print("DATA LOADING done.")
    print("=========================================================")
    print(f"loader: {loader}")

    tb_writer = SummaryWriter(log_dir=args.save_path +
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
        OVERALL_STEP=args.overall_step,
    )

    print("EVERYTHING in the TRAINING now DONE.")


if __name__ == "__main__":
    main()
