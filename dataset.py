import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NLUDataset(Dataset):
    def __init__(self, inputs, slots, intents):
        self.inputs = inputs
        self.slots = slots
        self.intents = intents

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "slot_labels": torch.tensor(self.slots[idx], dtype=torch.long),
            "intent_label": torch.tensor(self.intents[idx], dtype=torch.long),
        }

def pad_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    slot_labels = [item["slot_labels"] for item in batch]
    intent_labels = [item["intent_label"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    slot_labels_padded = pad_sequence(slot_labels, batch_first=True, padding_value=0)
    intent_labels = torch.stack(intent_labels)

    return {
        "input_ids": input_ids_padded,
        "slot_labels": slot_labels_padded,
        "intent_labels": intent_labels
    }
