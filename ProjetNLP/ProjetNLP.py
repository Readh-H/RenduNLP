import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

from utils import load_nlu_data_from_files, build_vocab, encode_sequences, encode_labels
from dataset import NLUDataset, pad_collate_fn
from models.model_bilstm import BiLSTM_NLU
from torch.utils.data import DataLoader

# ==== 1. Chargement des données d'entrainement ====
train_sentences, train_slots, train_intents = load_nlu_data_from_files(
    "ProjetNLP/data/train/slot-filling.in",
    "ProjetNLP/data/train/slot-filling.out",
    "ProjetNLP/data/train/intentlabels"
)

# ==== 2. Vocabulaire & encodage ====
word2idx = build_vocab(train_sentences)
slot2idx = build_vocab(train_slots, add_pad_unk=False)
slot2idx["<PAD>"] = slot2idx.get("<PAD>", len(slot2idx))
intent2idx = build_vocab([[intent] for intent in train_intents], add_pad_unk=False)

encoded_sentences = encode_sequences(train_sentences, word2idx)
encoded_slots = encode_labels(train_slots, slot2idx)
encoded_intents = [intent2idx[intent] for intent in train_intents]

# ==== 3. Dataset et Dataloader ====
train_dataset = NLUDataset(encoded_sentences, encoded_slots, encoded_intents)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

# ==== 4. Modèle ====
vocab_size = len(word2idx)
slot_size = len(slot2idx)
intent_size = len(intent2idx)
embedding_dim = 128
hidden_dim = 64
padding_idx = word2idx["<PAD>"]

model = BiLSTM_NLU(vocab_size, embedding_dim, hidden_dim, slot_size, intent_size, padding_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==== 5. Loss ====
def compute_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot_pad_idx):
    slot_logits = slot_logits.view(-1, slot_logits.shape[-1])
    slot_labels = slot_labels.view(-1)
    slot_loss = F.cross_entropy(slot_logits, slot_labels, ignore_index=slot_pad_idx)
    intent_loss = F.cross_entropy(intent_logits, intent_labels)
    return slot_loss + intent_loss, slot_loss.item(), intent_loss.item()

# ==== 6. Fonction d'entraînement ====
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_slot_loss, total_intent_loss = 0, 0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        slot_labels = batch["slot_labels"].to(device)
        intent_labels = batch["intent_labels"].to(device)

        optimizer.zero_grad()
        slot_logits, intent_logits = model(input_ids)
        loss, slot_l, intent_l = compute_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot2idx["<PAD>"])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_slot_loss += slot_l
        total_intent_loss += intent_l

    return total_loss / len(dataloader), total_slot_loss / len(dataloader), total_intent_loss / len(dataloader)

# ==== 7. Suivi des métriques ====
def compute_metrics_on_batch(model, loader, device, slot2idx, idx2slot):
    model.eval()
    intents_true, intents_pred = [], []
    slots_true, slots_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_labels = batch["intent_labels"].to(device)

            slot_logits, intent_logits = model(input_ids)
            pred_intents = torch.argmax(intent_logits, dim=1)
            intents_true.extend(intent_labels.cpu().tolist())
            intents_pred.extend(pred_intents.cpu().tolist())

            pred_slots = torch.argmax(slot_logits, dim=-1)
            for i in range(input_ids.size(0)):
                gold_seq, pred_seq = [], []
                for j in range(input_ids.size(1)):
                    gold = slot_labels[i][j].item()
                    pred = pred_slots[i][j].item()
                    if gold != slot2idx["<PAD>"]:
                        gold_seq.append(idx2slot[gold])
                        pred_seq.append(idx2slot[pred])
                slots_true.append(gold_seq)
                slots_pred.append(pred_seq)

    acc = accuracy_score(intents_true, intents_pred)
    f1 = f1_score(slots_true, slots_pred)
    return acc, f1

# ==== 8. Entraînement ====
train_loss_history = []
train_slot_loss_history = []
train_intent_loss_history = []
train_acc_history = []
train_f1_history = []

idx2slot = {v: k for k, v in slot2idx.items()}
for epoch in range(1, 4):
    print(f"\n🔁 Époque {epoch}")
    loss, slot_l, intent_l = train_epoch(model, train_loader, optimizer, device)
    print(f"Loss totale     : {loss:.4f}")
    print(f"Loss Slot       : {slot_l:.4f}")
    print(f"Loss Intention  : {intent_l:.4f}")

    train_loss_history.append(loss)
    train_slot_loss_history.append(slot_l)
    train_intent_loss_history.append(intent_l)

    acc, f1 = compute_metrics_on_batch(model, train_loader, device, slot2idx, idx2slot)
    train_acc_history.append(acc)
    train_f1_history.append(f1)

# ==== 9. Évaluation ====
idx2intent = {v: k for k, v in intent2idx.items()}
def evaluate(model, dataloader, device, idx2slot, idx2intent):
    model.eval()
    all_intents_true, all_intents_pred = [], []
    all_slots_true, all_slots_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_labels = batch["intent_labels"].to(device)

            slot_logits, intent_logits = model(input_ids)
            pred_intents = torch.argmax(intent_logits, dim=1)
            all_intents_true.extend(intent_labels.cpu().tolist())
            all_intents_pred.extend(pred_intents.cpu().tolist())

            pred_slots = torch.argmax(slot_logits, dim=-1)
            for i in range(len(input_ids)):
                true_seq, pred_seq = [], []
                for j in range(input_ids.shape[1]):
                    true_label = slot_labels[i, j].item()
                    pred_label = pred_slots[i, j].item()
                    if true_label != slot2idx["<PAD>"]:
                        true_seq.append(idx2slot[true_label])
                        pred_seq.append(idx2slot[pred_label])
                all_slots_true.append(true_seq)
                all_slots_pred.append(pred_seq)

    acc = accuracy_score(all_intents_true, all_intents_pred)
    precision = precision_score(all_slots_true, all_slots_pred)
    recall = recall_score(all_slots_true, all_slots_pred)
    f1 = f1_score(all_slots_true, all_slots_pred)

    print("\n🎯 Evaluation")
    print(f"Intent Accuracy     : {acc:.4f}")
    print(f"Slot Precision      : {precision:.4f}")
    print(f"Slot Recall         : {recall:.4f}")
    print(f"Slot F1-score       : {f1:.4f}")

evaluate(model, train_loader, device, idx2slot, idx2intent)

# ==== 10. Sauvegarde du modèle ====
os.makedirs("ProjetNLP/models", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "word2idx": word2idx,
    "slot2idx": slot2idx,
    "intent2idx": intent2idx
}, "ProjetNLP/models/bilstm_nlu.pt")
print("\n✅ Modèle sauvegardé dans : ProjetNLP/models/bilstm_nlu.pt")

# ==== 11. Chargement et évaluation test ====
test_sentences, test_slots, test_intents = load_nlu_data_from_files(
    "ProjetNLP/data/test/slot-filling.in",
    "ProjetNLP/data/test/slot-filling.out",
    "ProjetNLP/data/test/intentlabels"
)
encoded_test_sentences = encode_sequences(test_sentences, word2idx)
encoded_test_slots = encode_labels(test_slots, slot2idx)
encoded_test_intents = [intent2idx.get(i, 0) for i in test_intents]

test_dataset = NLUDataset(encoded_test_sentences, encoded_test_slots, encoded_test_intents)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)

print("\n🧪 Évaluation sur le jeu de test")
evaluate(model, test_loader, device, idx2slot, idx2intent)

# ==== 12. Génération des graphiques ====
os.makedirs("ProjetNLP/figures", exist_ok=True)
epochs = list(range(1, len(train_loss_history) + 1))

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_acc_history, marker='o', label="Intent Accuracy")
plt.plot(epochs, train_f1_history, marker='o', label="Slot F1-score")
plt.title("Évolution des scores - Entraînement")
plt.xlabel("Époque")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ProjetNLP/figures/metrics_scores.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss_history, marker='o', label="Loss totale")
plt.plot(epochs, train_slot_loss_history, marker='o', label="Slot Loss")
plt.plot(epochs, train_intent_loss_history, marker='o', label="Intent Loss")
plt.title("Évolution des pertes - Entraînement")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ProjetNLP/figures/metrics_losses.png")
plt.show()
