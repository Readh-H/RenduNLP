
"""
Script pour l'entraînement et la sauvegarde du modèle BiLSTM pour NLU

"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

# Définir les chemins des données
DATA_DIR = "data"  # Dossier principal des données
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Créer les dossiers s'ils n'existent pas
os.makedirs("models", exist_ok=True)

# Vérifier si les données sont présentes
def check_data_files():
    required_files = [
        os.path.join(TRAIN_DIR, "slot-filling.in"),
        os.path.join(TRAIN_DIR, "slot-filling.out"),
        os.path.join(TRAIN_DIR, "intentlabels"),
        os.path.join(TEST_DIR, "slot-filling.in"),
        os.path.join(TEST_DIR, "slot-filling.out"),
        os.path.join(TEST_DIR, "intentlabels")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ Fichiers manquants:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nVeuillez placer les fichiers de données dans les dossiers appropriés.")
        return False
    
    return True

# Fonctions utilitaires pour le chargement et le traitement des données
def load_nlu_data_from_files(text_file, slot_file, intent_file):
    """Charge les données NLU à partir des fichiers."""
    with open(text_file, encoding='utf-8') as f:
        sentences = [line.strip().split() for line in f]
    
    with open(slot_file, encoding='utf-8') as f:
        slots = [line.strip().split() for line in f]
    
    with open(intent_file, encoding='utf-8') as f:
        intents = [line.strip() for line in f]
    
    # Vérifier que les données sont cohérentes
    assert len(sentences) == len(slots) == len(intents), "Incohérence dans le nombre d'exemples"
    
    # Supprimer l'entête si présente
    if sentences[0][0].startswith('BOS'):
        print("🧹 Suppression de l'entête parasite...")
        sentences = sentences[1:]
        slots = slots[1:]
        intents = intents[1:]
    
    return sentences, slots, intents

def build_vocab(sequences, add_pad_unk=True):
    """Construit un vocabulaire à partir des séquences."""
    vocab = {}
    
    if add_pad_unk:
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
    
    for sequence in sequences:
        for token in sequence:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    return vocab

def encode_sequences(sequences, word2idx):
    """Encode les séquences de mots en utilisant le vocabulaire."""
    encoded = []
    for sequence in sequences:
        encoded.append([word2idx.get(token, word2idx["<UNK>"]) for token in sequence])
    return encoded

def encode_labels(sequences, label2idx):
    """Encode les séquences d'étiquettes en utilisant le vocabulaire."""
    encoded = []
    for sequence in sequences:
        encoded.append([label2idx[token] for token in sequence])
    return encoded

# Classe Dataset pour PyTorch
class NLUDataset(Dataset):
    def __init__(self, input_ids, slot_labels, intent_labels):
        self.input_ids = input_ids
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "slot_labels": torch.tensor(self.slot_labels[idx], dtype=torch.long),
            "intent_labels": torch.tensor(self.intent_labels[idx], dtype=torch.long)
        }

# Fonction de collate pour le padding
def pad_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    slot_labels = [item["slot_labels"] for item in batch]
    intent_labels = torch.tensor([item["intent_labels"] for item in batch])
    
    # Trouver la longueur maximale
    max_len = max(len(ids) for ids in input_ids)
    
    # Padding
    padded_input_ids = []
    padded_slot_labels = []
    
    for ids, labels in zip(input_ids, slot_labels):
        padding_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)]))
        padded_slot_labels.append(torch.cat([labels, torch.zeros(padding_len, dtype=torch.long)]))
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "slot_labels": torch.stack(padded_slot_labels),
        "intent_labels": intent_labels
    }

# Modèle BiLSTM pour NLU
class BiLSTM_NLU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, slot_size, intent_size, padding_idx=0):
        super(BiLSTM_NLU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Couche pour la classification des slots
        self.slot_classifier = nn.Linear(hidden_dim * 2, slot_size)
        
        # Couche pour la classification des intentions
        self.intent_classifier = nn.Linear(hidden_dim * 2, intent_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM
        lstm_out, (hidden, _) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim*2]
        
        # Classification des slots
        slot_logits = self.slot_classifier(lstm_out)  # [batch_size, seq_len, slot_size]
        
        # Classification des intentions (utiliser la dernière sortie cachée)
        intent_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_dim*2]
        intent_logits = self.intent_classifier(intent_hidden)  # [batch_size, intent_size]
        
        return slot_logits, intent_logits

# Fonction de calcul de la perte
def compute_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot_pad_idx):
    slot_logits = slot_logits.view(-1, slot_logits.shape[-1])
    slot_labels = slot_labels.view(-1)
    slot_loss = F.cross_entropy(slot_logits, slot_labels, ignore_index=slot_pad_idx)
    intent_loss = F.cross_entropy(intent_logits, intent_labels)
    return slot_loss + intent_loss, slot_loss.item(), intent_loss.item()

# Fonction d'entraînement pour une époque
def train_epoch(model, dataloader, optimizer, device, slot_pad_idx):
    model.train()
    total_loss, total_slot_loss, total_intent_loss = 0, 0, 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        slot_labels = batch["slot_labels"].to(device)
        intent_labels = batch["intent_labels"].to(device)
        
        optimizer.zero_grad()
        slot_logits, intent_logits = model(input_ids)
        loss, slot_l, intent_l = compute_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot_pad_idx)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_slot_loss += slot_l
        total_intent_loss += intent_l
    
    return total_loss / len(dataloader), total_slot_loss / len(dataloader), total_intent_loss / len(dataloader)

# Fonction de calcul des métriques
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

# Fonction d'évaluation complète
def evaluate(model, dataloader, device, slot2idx, idx2slot, idx2intent):
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
    
    intent_acc = accuracy_score(all_intents_true, all_intents_pred)
    slot_precision = precision_score(all_slots_true, all_slots_pred)
    slot_recall = recall_score(all_slots_true, all_slots_pred)
    slot_f1 = f1_score(all_slots_true, all_slots_pred)
    
    print("\n🎯 Evaluation")
    print(f"Intent Accuracy     : {intent_acc:.4f}")
    print(f"Slot Precision      : {slot_precision:.4f}")
    print(f"Slot Recall         : {slot_recall:.4f}")
    print(f"Slot F1-score       : {slot_f1:.4f}")
    
    return {
        "intent_accuracy": intent_acc,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1
    }

# Fonction principale pour l'entraînement du modèle BiLSTM
def train_bilstm():
    print("\n" + "="*50)
    print("🚀 Entraînement du modèle BiLSTM")
    print("="*50)
    
    # Mesurer le temps d'entraînement
    start_time = time.time()
    
    # Chargement des données d'entraînement
    train_sentences, train_slots, train_intents = load_nlu_data_from_files(
        os.path.join(TRAIN_DIR, "slot-filling.in"),
        os.path.join(TRAIN_DIR, "slot-filling.out"),
        os.path.join(TRAIN_DIR, "intentlabels")
    )
    
    print(f"Nombre de phrases   : {len(train_sentences)}")
    print(f"Nombre de slots     : {len(train_slots)}")
    print(f"Nombre d'intentions : {len(train_intents)}")
    
    # Vocabulaire & encodage
    word2idx = build_vocab(train_sentences)
    slot2idx = build_vocab(train_slots, add_pad_unk=False)
    slot2idx["<PAD>"] = slot2idx.get("<PAD>", len(slot2idx))
    intent2idx = build_vocab([[intent] for intent in train_intents], add_pad_unk=False)
    
    encoded_sentences = encode_sequences(train_sentences, word2idx)
    encoded_slots = encode_labels(train_slots, slot2idx)
    encoded_intents = [intent2idx[intent] for intent in train_intents]
    
    # Dataset et Dataloader
    train_dataset = NLUDataset(encoded_sentences, encoded_slots, encoded_intents)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)
    
    # Modèle
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
    
    # Entraînement
    idx2slot = {v: k for k, v in slot2idx.items()}
    idx2intent = {v: k for k, v in intent2idx.items()}
    
    for epoch in range(1, 4):
        print(f"\n🔁 Époque {epoch}")
        loss, slot_l, intent_l = train_epoch(model, train_loader, optimizer, device, slot2idx["<PAD>"])
        print(f"Loss totale     : {loss:.4f}")
        print(f"Loss Slot       : {slot_l:.4f}")
        print(f"Loss Intention  : {intent_l:.4f}")
        
        acc, f1 = compute_metrics_on_batch(model, train_loader, device, slot2idx, idx2slot)
        print(f"Intent Accuracy : {acc:.4f}")
        print(f"Slot F1-score   : {f1:.4f}")
    
    # Évaluation
    train_results = evaluate(model, train_loader, device, slot2idx, idx2slot, idx2intent)
    
    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
        "slot2idx": slot2idx,
        "intent2idx": intent2idx
    }, "models/bilstm_nlu.pt")
    print("\n✅ Modèle sauvegardé dans : models/bilstm_nlu.pt")
    
    # Chargement et évaluation test
    test_sentences, test_slots, test_intents = load_nlu_data_from_files(
        os.path.join(TEST_DIR, "slot-filling.in"),
        os.path.join(TEST_DIR, "slot-filling.out"),
        os.path.join(TEST_DIR, "intentlabels")
    )
    
    print(f"Nombre de phrases   : {len(test_sentences)}")
    print(f"Nombre de slots     : {len(test_slots)}")
    print(f"Nombre d'intentions : {len(test_intents)}")
    
    encoded_test_sentences = encode_sequences(test_sentences, word2idx)
    encoded_test_slots = encode_labels(test_slots, slot2idx)
    encoded_test_intents = [intent2idx.get(i, 0) for i in test_intents]
    
    test_dataset = NLUDataset(encoded_test_sentences, encoded_test_slots, encoded_test_intents)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)
    
    print("\n🧪 Évaluation sur le jeu de test")
    test_results = evaluate(model, test_loader, device, slot2idx, idx2slot, idx2intent)
    
    # Afficher le temps d'entraînement
    training_time = time.time() - start_time
    print(f"\n⏱️ Temps d'entraînement : {training_time:.2f} secondes")
    
    return model, test_results

# Fonction principale
def main():
    print("🔍 Vérification des données...")
    if not check_data_files():
        return
    
    # Entraînement du modèle BiLSTM
    model, results = train_bilstm()
    
    print("\n✅ Entraînement terminé avec succès!")
    print("📊 Résultats sur le jeu de test:")
    print(f"Intent Accuracy : {results['intent_accuracy']:.4f}")
    print(f"Slot Precision  : {results['slot_precision']:.4f}")
    print(f"Slot Recall     : {results['slot_recall']:.4f}")
    print(f"Slot F1-score   : {results['slot_f1']:.4f}")

if __name__ == "__main__":
    main()
