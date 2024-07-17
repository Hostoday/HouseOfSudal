import argparse
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup,
                         get_linear_schedule_with_warmup)
from src.ed_model_data import CustomDataset, DataCollatorForSupervisedDataset
from src.utils import set_random_seed
from src.arg_parser import get_args
from accelerate import Accelerator

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score
import wandb
import logging
from datetime import datetime, timezone, timedelta

from datasets import Dataset , load_metric

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

def init_model(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model

def init_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_id)
    tokenizer.padding_side = "right"

    return tokenizer

# Function to calculate ROUGE score
def calculate_rouge(true, pred):
    rouge_evaluator = Rouge()
    scores = rouge_evaluator.get_scores(pred, true, avg=True)
    return scores['rouge-1']['f']

# Function to calculate BERTScore
def calculate_bertscore(true, pred):
    P, R, F1 = bert_score(cands=pred, refs=true, lang="ko", model_type='bert-base-multilingual-cased', rescale_with_baseline=True)
    return F1.mean().item()

# Function to calculate BLEURT score
#def calculate_bleurt(references, candidates, bleurt_checkpoint='BLEURT-20'):
#    scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
#    scores = scorer.score(references=references, candidates=candidates)
#    return scores

def metrix_score(references, candidates):
    rouge = calculate_rouge(references,candidates)
    bertscore = calculate_bertscore(references,candidates)

    avg_score = (rouge+bertscore)/2
    return avg_score


def main(args):
    accelerate = Accelerator()

    tokenizer = init_tokenizer(args)
    set_random_seed(args.seed)
    
    train_data = CustomDataset("/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/data/일상대화요약_train.json",tokenizer)
    eval_data = CustomDataset("/home/nlplab/ssd1/jeong/Korean_DCS_2024/baseline/Korean_DCS_2024/resource/data/일상대화요약_dev.json",tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True , collate_fn=DataCollatorForSupervisedDataset(tokenizer),batch_size = args.batch_size)
    eval_dataloader = torch.utils.data.DataLoader(eval_data, shuffle=False, collate_fn=DataCollatorForSupervisedDataset(tokenizer), batch_size = args.batch_size)

    model = init_model(args)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_dataloader) * args.epoch // args.gradient_accumulation_steps
    scheduler = (get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps) 
                 if args.scheduler_type == "cosine" 
                 else get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps))
    
    train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerate.prepare(train_dataloader,eval_dataloader, model, optimizer, scheduler)

    step = 0

    for ep in range(args.epoch):
        model.train()
        print(f"Epoch 시작: {ep+1}")
        for i, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            attention_masks = batch["attention_masks"]
            labels = batch["labels"]
            
            output = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels).logits
            loss = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels).loss/args.gradient_accumulation_steps
            accelerate.backward(loss)
            step += 1
            
            if step % args.gradient_accumulation_steps == 0: 
                print(f"Loss: {loss.item()}, Step: {step}")
                wandb.log({"loss": loss.item(), "step": step})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        wandb.log({"epoch": ep+1})
        epoch_save_dir = os.path.join(args.save_dir, f"{args.model_id}/{args.wandb_run_name}/train_{ep+1}")
        if not os.path.exists(epoch_save_dir):
            os.makedirs(epoch_save_dir)

        model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        with open(os.path.join(epoch_save_dir, 'training_args.bin'), 'wb') as f:
            torch.save(args, f)
        
        model.eval()
        generated_texts = []
        labels_list = []

        for i,batch in enumerate(tqdm(eval_dataloader)):
            input_ids = batch["input_ids"]
            attention_masks = batch["attention_masks"]
            labels = batch["labels"]

            outputs = model.generate(
                input_ids = input_ids,
                attention_mask = attention_masks,
                min_length = args.min_length,
                max_length = args.max_length,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                do_sample = True,
                top_k = args.top_k,
                top_p = args.top_p,
                early_stopping = True
                                    )
            for label in labels:
                summary = tokenizer.decode(label,skip_special_tokens = True)
                labels_list.append(summary)

            for output in outputs:
                text = tokenizer.decode(output,skip_special_tokens = True)
                generated_texts.append(text)

        for i, text in enumerate(generated_texts):
            text = text.replace("<pad>","")
            text = text.replace("</s>","")

        for i, label in enumerate(labels_list):
            label = label.replace("<pad>","")
            label = label.replace("</s>","")
        
        score = metrix_score(labels_list,generated_texts)
        print(f"Evaulation Score : {score}")
        wandb.log({'Evaluated Score' : score})


if __name__ == "__main__":
    args = get_args()

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=args,
    )
    wandb.config.update(args)

    exit(main(args))
        
        
            


            
            

