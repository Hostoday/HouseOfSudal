
import argparse
import json
from tqdm import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset
from peft import PeftModel, PeftConfig


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--batch_size", type=int, default=4, help="batch size")
# fmt: on


def main(args):

    config = PeftConfig.from_pretrained(args.model_ckpt_path)
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
        device_map=args.device,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, args.model_ckpt_path)
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CustomDataset("resource/data/일상대화요약_test.json", tokenizer)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )

    with open("resource/data/일상대화요약_test.json", "r") as f:
        result = json.load(f)

    batch_start_idx = 0
    for batch in tqdm(test_dataloader, desc="Test"):
        inp = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        outputs = model.generate(
            inp,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            early_stopping=True
        )

        generated_texts = [tokenizer.decode(output[inp.shape[-1]:], skip_special_tokens=True) for output in outputs]
        print(generated_texts)

        for i, text in enumerate(generated_texts):
            result[batch_start_idx + i]["output"] = text

        batch_start_idx += len(generated_texts)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))