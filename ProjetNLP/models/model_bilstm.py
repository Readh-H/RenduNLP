import torch
import torch.nn as nn

class BiLSTM_NLU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, slot_size, intent_size, padding_idx=0):
        super(BiLSTM_NLU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.slot_classifier = nn.Linear(hidden_dim * 2, slot_size)
        self.intent_classifier = nn.Linear(hidden_dim * 2, intent_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)                           # [B, T, E]
        lstm_out, (h_n, c_n) = self.bilstm(embedded)                   # lstm_out: [B, T, 2H]

        # Slot filling: un tag par token
        slot_logits = self.slot_classifier(lstm_out)                  # [B, T, slot_size]

        # Intent detection: basé sur h_n concat (last hidden state des 2 directions)
        h_forward = h_n[0, :, :]  # [B, H]
        h_backward = h_n[1, :, :] # [B, H]
        h_concat = torch.cat((h_forward, h_backward), dim=1)          # [B, 2H]
        intent_logits = self.intent_classifier(h_concat)              # [B, intent_size]

        return slot_logits, intent_logits
