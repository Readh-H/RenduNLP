import torch
import torch.nn as nn
from transformers import CamembertModel

class CamembertNLU(nn.Module):
    def __init__(self, slot_size, intent_size):
        super().__init__()
        self.encoder = CamembertModel.from_pretrained("camembert-base")

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.slot_classifier = nn.Linear(hidden_size, slot_size)
        self.intent_classifier = nn.Linear(hidden_size, intent_size)

    def forward(self, input_ids, attention_mask, slot_labels=None, intent_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state      # [B, T, H]
        pooled_output = outputs.pooler_output            # [B, H]

        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        slot_logits = self.slot_classifier(sequence_output)  # [B, T, slot_size]
        intent_logits = self.intent_classifier(pooled_output)  # [B, intent_size]

        loss = None
        if slot_labels is not None and intent_labels is not None:
            loss_fct_slot = nn.CrossEntropyLoss(ignore_index=-100)
            loss_fct_intent = nn.CrossEntropyLoss()
            # slot_logits: [B, T, slot_size] -> [B*T, slot_size], slot_labels: [B, T] -> [B*T]
            slot_loss = loss_fct_slot(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
            intent_loss = loss_fct_intent(intent_logits, intent_labels)
            loss = slot_loss + intent_loss
            return loss, slot_logits, intent_logits

        return slot_logits, intent_logits
