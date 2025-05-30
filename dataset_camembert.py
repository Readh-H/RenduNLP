from torch.utils.data import Dataset
from transformers import CamembertTokenizerFast
import torch
import json

class CamembertNLUDataset(Dataset):
    def __init__(self, json_path, slot2idx, intent2idx, max_len=32):  # <-- Mets 32 ou moins
        self.tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
        self.max_len = max_len
        self.slot2idx = slot2idx
        self.intent2idx = intent2idx

        with open(json_path, encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["tokens"]
        slots = item["slots"]
        intent = item["intent"]

        # Tokenization avec alignement
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        slot_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                slot_labels.append(-100)  # ignoré par la loss
            elif word_idx != previous_word_idx:
                slot_labels.append(self.slot2idx.get(slots[word_idx], 0))
            else:
                label = slots[word_idx]
                if label.startswith("B-"):
                    label = label.replace("B-", "I-")
                slot_labels.append(self.slot2idx.get(label, 0))
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "slot_labels": torch.tensor(slot_labels),
            "intent_labels": torch.tensor(self.intent2idx[intent]) 
        }
