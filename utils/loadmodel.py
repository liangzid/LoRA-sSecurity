"""
======================================================================
LOADMODEL --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 16 August 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

from transformers import AutoModel




base_ls=["google-bert/bert-large-uncased",
         "FacebookAI/roberta-large",
         "microsoft/deberta-v3-large"]

for base in base_ls:
    print(f"load {base}")
    a=AutoModel.from_pretrained(base,
                                device_map="cuda",
                                )


    
