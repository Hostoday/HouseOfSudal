import argparse
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
                          Trainer, TrainingArguments, get_cosine_schedule_with_warmup, 
                          get_linear_schedule_with_warmup)
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from src.utils import set_random_seed
from src.arg_parser import get_args
from accelerate import Accelerator
import wandb

from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
from datetime import datetime, timezone, timedelta

from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomCallback(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Starting epoch {state.epoch}...")
        print(f"Starting epoch {state.epoch}...")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Log: {state.log_history[-1]}")
        print(f"Log: {state.log_history[-1]}")

def init_model(args):
    """Initialize the model with 4-bit quantization and LoRA configuration."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        quantization_config=quantization_config,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, peft_config

def init_tokenizer(args):
    """Initialize the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def loss_fn(target, outputs, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    logits = outputs.logits.view(-1, outputs.logits.size(-1))
    target = target.view(-1)
    loss = criterion(logits, target)
    return loss

def train_model(args):
    model, peft_config = init_model(args)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    set_random_seed(args.seed)
    tokenizer = init_tokenizer(args)

    train_dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({'input_ids': train_dataset.inp, 'labels': train_dataset.label})
    valid_dataset = Dataset.from_dict({'input_ids': valid_dataset.inp, 'labels': valid_dataset.label})
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        save_strategy="epoch",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        do_train=True,
        do_eval=False,
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=f'{args.save_dir}/{args.model_id}/{datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M")}',
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[CustomCallback()],
    )
    
    trainer.train()
    trainer.save_model()

def main(args):
    accelerator = Accelerator()
    model, peft_config = init_model(args)
    set_random_seed(args.seed)
    tokenizer = init_tokenizer(args)

    train_dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer)
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer)
    )

    total_steps = len(train_dataloader) * args.epoch // args.gradient_accumulation_steps
    scheduler = (get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps) 
                 if args.scheduler_type == "cosine" 
                 else get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps))
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    step = 0
    best_eval_loss = float('inf')
    saved_models = []

    for ep in range(args.epoch):
        model.train()
        print(f"Epoch {ep + 1} 시작")
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {ep+1}")):
            inputs = batch["input_ids"]
            labels = batch["labels"]
            outputs = model(inputs)
            loss = loss_fn(labels, outputs, tokenizer) / args.gradient_accumulation_steps
            accelerator.backward(loss)
            step += 1
            
            if step % args.gradient_accumulation_steps == 0:
                print(f"Loss: {loss.item()}, Step: {step}")
                wandb.log({"loss": loss.item(), "step": step})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        wandb.log({"epoch": ep+1})
        
        epoch_save_dir = os.path.join(args.save_dir, f"{ep+1}")
        if not os.path.exists(epoch_save_dir):
            os.makedirs(epoch_save_dir)

        model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        with open(os.path.join(epoch_save_dir, 'training_args.bin'), 'wb') as f:
            torch.save(args, f)    
            
        print(f"Epoch {ep + 1} 끝")

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_dataloader, desc=f"Validation Epoch {ep+1}")):
                inputs = batch["input_ids"]
                labels = batch["labels"]
                outputs = model(inputs)
                loss = loss_fn(labels, outputs, tokenizer) / args.gradient_accumulation_steps
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(valid_dataloader)
        print(f"Validation Loss after Epoch {ep+1}: {avg_eval_loss}")
        wandb.log({"eval_loss": avg_eval_loss})

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_path = args.save_dir + f"/epoch_{ep+1}_loss_{avg_eval_loss:.4f}"
            model.save_pretrained(save_path)
            saved_models.append(save_path)
            
            if len(saved_models) > args.save_total_limit:
                model_to_remove = saved_models.pop(0)
                if os.path.exists(model_to_remove):
                    os.system(f"rm -rf {model_to_remove}")

if __name__ == "__main__":
    args = get_args()

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=args,
    )
    wandb.config.update(args)
    
    if args.trainer == "True":
        exit(train_model(args)) 
    else:
        exit(main(args))
