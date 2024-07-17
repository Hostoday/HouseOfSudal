import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, prompt, mode='mode_with_special_tokens_concat'):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.speaker_mappings = []
        
        PROMPT = prompt


        with open(fname, "r") as f:
            data = json.load(f)
            
        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)

        def make_chat_with_special_tokens_concat(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        def make_chat_with_special_tokens(inp, speaker_map):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                chat.append(f"{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def make_chat_with_special_tokens_topic_intro(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}가 {utterance}라고 말했다.")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        def make_chat_original(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        def process_target_with_special_tokens(target, speaker_map, tokenizer):
            for speaker in speaker_map:
                target = target.replace(speaker, speaker_map[speaker])
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")

        def process_target_original(target, tokenizer):
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")
        
        for example in data:
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            
            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}

            if mode == 'mode_with_special_tokens_concat':
                chat = make_chat_with_special_tokens_concat(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_with_special_tokens':
                chat = make_chat_with_special_tokens(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_with_special_tokens_topic_intro':
                chat = make_chat_with_special_tokens_topic_intro(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_original':
                chat = make_chat_original(example["input"])
                process_target_func = process_target_original
            
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token

            target = process_target_func(target, speaker_map if mode != 'mode_original' else tokenizer, tokenizer)
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.speaker_mappings.append(speaker_map)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "labels": self.label[idx], "speaker_map": self.speaker_mappings[idx]}

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels, speaker_maps = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "speaker_map"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'speaker_maps': speaker_maps
        }