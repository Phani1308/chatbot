#!/usr/bin/env python
# coding: utf-8

# ## Installing Necessary Libraries

# In[ ]:





# # 

# In[2]:


get_ipython().system('pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git')
get_ipython().system('pip install -q datasets bitsandbytes einops wandb')


# # Dataset details
# Instacart Data can be downloaded from [here](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data). We just need product & department csv files
# 

# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system("ls /content/drive/MyDrive/'Colab Notebooks'")


# In[ ]:


import pandas as pd
df_product = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/products.csv")
df_dept = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/departments.csv')


# In[ ]:


df_joined = pd.merge(df_product, df_dept, on = ['department_id'])
df_joined['text'] = df_joined.apply(lambda row: row['product_name'] + " ->: " + row['department'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_joined, test_size=0.2, random_state=42)


# In[ ]:


train_df.head(10)


# In[ ]:


test_df.head(10)


# In[ ]:


from datasets import Dataset,DatasetDict
train_dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df),
})


# In[ ]:


train_dataset_dict


# ## Loading the model

# In this section we will load the [Falcon 7B model](https://huggingface.co/tiiuae/falcon-7b), quantize it in 4bit and attach LoRA adapters on it. Let's get started!

# In[ ]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "ybelkada/falcon-7b-sharded-bf16"
# model_name = "tiiuae/falcon-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False


# Let's also load the tokenizer below

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# **Let's check what the base model predicts before finetuning. :)**

# In[ ]:


import transformers
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


sequences = pipeline(
   ["“Free & Clear Stage 4 Overnight Diapers” ->:","Bread Rolls ->:","French Milled Oval Almond Gourmande Soap ->:"],
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq[0]['generated_text']}")


# Below we will load the configuration file in order to create the LoRA model. According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `dense`, `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.

# In[ ]:


from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)


# ## Loading the trainer

# Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# In[ ]:


from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 120 #500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


# Then finally pass everthing to the trainer

# In[ ]:


from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_dict['train'],
    # train_dataset=data['train'],
    peft_config=peft_config,
    dataset_text_field="text",
    # dataset_text_field="prediction",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training

# In[ ]:


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


# ## Train the model

# Now let's train the model! Simply call `trainer.train()`

# In[ ]:


trainer.train()


# In[ ]:


lst_test_data = list(test_df['text'])


# In[ ]:


len(lst_test_data)


# In[ ]:


sample_size = 25
lst_test_data_short = lst_test_data[:sample_size]


# In[ ]:


import transformers

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)

sequences = pipeline(
    lst_test_data_short,
    max_length=100,  #200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

for ix,seq in enumerate(sequences):
    print(ix,seq[0]['generated_text'])


# In[ ]:


def correct_answer(ans):
  return (ans.split("->:")[1]).strip()

answers = []
for ix,seq in enumerate(sequences):
    # print(ix,seq[0]['generated_text'])
    answers.append(correct_answer(seq[0]['generated_text']))

answers


# In[ ]:


df_evaluate = test_df.iloc[:sample_size][['product_name','department']]

df_evaluate = df_evaluate.reset_index(drop=True)

df_evaluate['department_predicted'] = answers

df_evaluate


# In[ ]:




