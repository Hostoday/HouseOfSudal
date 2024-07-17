
import argparse
import json
from tqdm import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset, DataCollatorForInferenceDataset
from peft import PeftModel, PeftConfig
import os

parser = argparse.ArgumentParser(prog="ed_model_inf", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--batch_size", type=int, default=2, help="batch size")
g.add_argument("--top_p", type = float, default = 0.94, help = "top-p inference")
g.add_argument("--top_k", type = int, default = 50, help = "top-k inference")
g.add_argument("--min_length",type = int, default = 82, help = "length minimize")


def init_model(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_ckpt_path)

    model.config.use_cache = False
    model.gradiant_checkpointing_enable()

    return model

def init_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_id)
    toeknizer.padding_side = "right"

    return tokenizer


def main(args):
    tokenizer = init_tokenizer()
    set_random_seed(args.seed)

    test_data = CustomDataset("resource/data/일상대화요약_test.json",toeknizer)
    test_dataloader = torch.utils.DataLoader(test_data, shuffle = False, collate_fn =DataCollatorForInferenceDataset(tokenizer),batch_size = args.batch_size )

    model = init_model()
    model.resize_token_embeddings(tokenizer)
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)

    with open("resource/data/일상대화요약_test.json", "r") as f:
        result = json.load(f)

    batch_start_idx = 0
    for i, batch in enumerate(test_dataloader):
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_masks"]
        speaker_map = batch["speaker_map"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            min_length=args.min_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            early_stopping=True
                                    )

        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output,skip_special_tokens = False)
            generated_texts.append(text)

        for i, text in enumerate(generated_texts):
            speaker_map = speakermap[i]
            for token, speaker in speaker_map.items():
                text = text.replace(speaker, token)
            text.replace("<pad>","")
            text.replace("</s>","")
            result[batch_start_idx + i]["output"] = text

        batch_start_idx += len(generated_texts)

    with open(f"inference/{args.output}", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))
    