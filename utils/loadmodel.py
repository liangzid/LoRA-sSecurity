

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


    
