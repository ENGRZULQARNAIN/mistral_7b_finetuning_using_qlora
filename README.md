
# Fine-Tuning Mistral 7B Model

This repository contains the process of fine-tuning the Mistral 7B model on a real-world problem. The notebook provides a comprehensive guide to installing dependencies, importing necessary libraries, and executing the fine-tuning process.

## Table of Contents
1. [Introduction](#introduction)
2. [Installing Dependencies](#installing-dependencies)
3. [Importing Libraries](#importing-libraries)
4. [Loading Dataset](#loading-dataset)
5. [Model and Tokenizer](#model-and-tokenizer)
6. [LoRA Configuration](#lora-configuration)
7. [Training Arguments](#training-arguments)
8. [Fine-Tuning](#fine-tuning)
9. [Evaluation](#evaluation)
10. [Inference](#inference)

## Introduction
This project demonstrates the fine-tuning of the Mistral 7B model using the `transformers`, `peft`, and `trl` libraries. Fine-tuning allows the pre-trained model to adapt to specific tasks by training it on a custom dataset.

## Installing Dependencies
To run the notebook, you need to install several dependencies:

```python
%%capture
%pip install accelerate peft bitsandbytes transformers trl
!pip install datasets nltk
```

## Importing Libraries
The necessary libraries are imported to handle dataset loading, model training, and evaluation.

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
```

## Loading Dataset
The dataset is loaded using the `datasets` library. You can replace the dataset with your custom dataset.

```python
dataset = load_dataset("yelp_review_full")
```

## Model and Tokenizer
Load the pre-trained Mistral 7B model and its corresponding tokenizer. 

```python
model_name = "mistralai/Mistral-7B"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## LoRA Configuration
Configure the Low-Rank Adaptation (LoRA) to reduce the number of trainable parameters, making the fine-tuning process more efficient.

```python
lora_alpha = 16
lora_config = LoraConfig(
    r=64,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Training Arguments
Set up the training arguments such as the learning rate, batch size, and logging details.

```python
output_dir = "./results"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    max_steps=300,
    logging_dir=f"{output_dir}/logs",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    dataloader_num_workers=3,
    group_by_length=True,
    report_to="tensorboard",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=5,
)
```

## Fine-Tuning
The `SFTTrainer` from the `trl` library is used to handle the fine-tuning process.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)
trainer.train()
model.save_pretrained(output_dir)
```

## Evaluation
Evaluate the fine-tuned model to check its performance.

```python
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")
```

## Inference
Generate predictions using the fine-tuned model.

```python
prompter = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompter("Explain fine-tuning in simple terms.")
```

## Conclusion
This notebook provides a detailed process for fine-tuning the Mistral 7B model on a custom dataset. By following these steps, you can adapt the model to various tasks, enhancing its performance on specific problems.
