import json
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score

from transformers import (
    CamembertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from torch.utils.data import random_split
from dataset_camembert import CamembertNLUDataset
from models.model_camembert import CamembertNLU

# === Lecture des données ===
with open("ProjetNLP/data/processed/train_camembert.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Extraction des labels uniques ===
all_slots = sorted({label for sample in raw_data for label in sample["slots"]})
all_intents = sorted({sample["intent"] for sample in raw_data})

slot2idx = {label: idx for idx, label in enumerate(all_slots)}
intent2idx = {label: idx for idx, label in enumerate(all_intents)}
idx2slot = {v: k for k, v in slot2idx.items()}
idx2intent = {v: k for k, v in intent2idx.items()}

# === Dataset complet
dataset = CamembertNLUDataset("ProjetNLP/data/processed/train_camembert.json", slot2idx, intent2idx)

# === Split train/valid
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# === Modèle
model = CamembertNLU(slot_size=len(slot2idx), intent_size=len(intent2idx))

# === Tokenizer + DataCollator
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

# === Métriques custom
def compute_metrics(p):
    predictions, labels = p
    slot_preds, intent_preds = predictions
    slot_labels, intent_labels = labels

    slot_preds = np.argmax(slot_preds, axis=2)

    true_slots = []
    pred_slots = []

    for pred, true in zip(slot_preds, slot_labels):
        true_seq, pred_seq = [], []
        for t, p in zip(true, pred):
            if t != -100:
                true_seq.append(idx2slot[t])
                pred_seq.append(idx2slot[p])
        true_slots.append(true_seq)
        pred_slots.append(pred_seq)

    intent_preds = np.argmax(intent_preds, axis=1)
    intent_acc = accuracy_score(intent_labels, intent_preds)
    slot_f1 = f1_score(true_slots, pred_slots)
    return {
        "intent_acc": intent_acc,
        "slot_f1": slot_f1,
        "slot_precision": precision_score(true_slots, pred_slots),
        "slot_recall": recall_score(true_slots, pred_slots),
    }

# === Entraînement
args = TrainingArguments(
    output_dir="ProjetNLP/models/camembert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,   # <-- Mets 2 ou même 1
    per_device_eval_batch_size=2,    # <-- Mets 2 ou même 1
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="ProjetNLP/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="slot_f1",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# === Sauvegarde
trainer.save_model("ProjetNLP/models/camembert_final")
print("✅ Modèle Camembert sauvegardé")
