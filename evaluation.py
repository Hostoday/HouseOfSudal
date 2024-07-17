import argparse
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForCausalLM,EarlyStoppingCallback,
                        AutoModelForSeq2SeqLM)

from src.utils import set_random_seed
from src.arg_parser import get_args
from accelerate import Accelerator
import wandb

from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
from datetime import datetime, timezone, timedelta

from datasets import Dataset , load_metric

import numpy as np

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--batch_size", type=int, default=2, help="batch size")
g.add_argument("--encoder", type=bool, default=False, help="batch size")
g.add_argument("--prompt",type=str,default = "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요.")


def main(args):

    config = PeftConfig.from_pretrained(args.model_ckpt_path)
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    if args.encoder is False:
        from src.data import CustomDataset, DataCollatorForInferenceDataset
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        quantization_config=quantization_config,
        )

        model.resize_token_embeddings(len(tokenizer))

    else:
        from src.encoder_data import CustomDataset, DataCollatorForInferenceDataset
        tokenizer.padding_side = "right"

        model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        #device_map=args.device,
        #quantization_config=quantization_config,
        )
        model.resize_token_embeddings(len(tokenizer))

    dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer, args.prompt)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForInferenceDataset(tokenizer),
    )
    
    
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)

    with open("resource/data/일상대화요약_dev.json", "r") as f:
        result = json.load(f)

    batch_start_idx = 0
    for batch in tqdm(test_dataloader, desc="Test"):
        inp = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        speakermap = batch["speaker_maps"]
        if args.encoder is False:
            outputs = model.generate(
                inp,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                early_stopping=True
            )

            generated_texts = []
            for output in outputs:
                text = tokenizer.decode(output[inp.shape[-1]:], skip_special_tokens=False)
                generated_texts.append(text)

            # Replace special tokens with speaker IDs
            for i, text in enumerate(generated_texts):
                speaker_map = speakermap[i]
                for token, speaker in speaker_map.items():
                    text = text.replace(speaker, token)
                text = text.replace("<|end_of_text|>", "")
                text = text.replace("<|begin_of_text|>", "")

        else:
            outputs = model.generate(
                inp,
                attention_mask = attention_mask,
                max_new_token = 1024,
                min_length = 82,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                do_sample = True,
                top_k = 100,
                top_p = 0.95,
                early_stopping = True
            )

            generated_texts = []
            for output in outputs:
                text = tokenizer.decode(output,skip_special_tokens = False)
                generated_texts.append(text)

            for i, text in enumerate(generated_texts):
                text = text.replace("<pad>","")
                text = text.replace("</s>","")


if __name__ == "__main__":
    exit(main(parser.parse_args()))