import json
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        self.inp = []
        self.attention = []
        self.label = []
        self.speaker_mappings = []

        with open(fname, "r") as f:
            data = json.load(f)
        
        #화자에 대한 special token 추가
        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)

        # Data prompt 한 것
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
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            
            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}
            chat = make_chat_with_special_tokens_topic_intro(example["input"], speaker_map)
            
            message = chat
            source = tokenizer(message, return_tensors="pt")
            target = example["output"]

            if target != "":
                target += tokenizer.eos_token

            target = tokenizer(target, return_tensors="pt")
            input_ids = source["input_ids"][0]
            attention_mask = source["attention_mask"][0]
            labels =  target["input_ids"][0]


            self.inp.append(input_ids)
            self.label.append(labels)
            self.attention.append(attention_mask)
            self.speaker_mappings.append(speaker_map)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "attention_masks" : self.attention[idx], "labels" : self.label[idx], "speaker_maps" : self.speaker_mappings[idx]}


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids,attention_masks, labels = tuple([instance[key] for instance in instances] for key in ("input_ids","attention_masks", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences=input_ids, batch_first=True)
        attention_masks = torch.nn.utils.rnn.pad_sequence(sequences=attention_masks,batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(sequences=labels, batch_first=True, padding_value=-100)
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_masks':attention_masks,
        }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids,attention_masks, speaker_maps = tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_masks", "speaker_map"))
        input_ids = torch.nn.rnn.pad_sequence(sequences=input_ids)
        attention_masks = torch.nn.rnn.pad_sequence(sequences=attention_masks)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'speaker_maps': speaker_maps
        }