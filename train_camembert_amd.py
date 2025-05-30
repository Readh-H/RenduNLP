import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Pour forcer certaines opérations sur GPU <-- echec

import torch
torch.set_num_threads(2)  # Réduire l'utilisation du CPU

import json
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score

print("Dossier courant :", os.getcwd())
print("Fichier existe :", os.path.exists("data/processed/train_camembert.json"))

# Vérification explicite de l'utilisation du GPU
import torch_directml
dml = torch_directml.device()
print(f"🖥️ Device DirectML : {dml}")
print(f"DirectML disponible : {torch_directml.is_available()}")

# Test explicite du GPU
try:
    # Créer un tenseur sur CPU
    x_cpu = torch.randn(1000, 1000)
    print("Tenseur CPU créé")
    
    # Déplacer sur GPU via DirectML
    x_gpu = x_cpu.to(dml)
    print("Tenseur déplacé sur GPU")
    
    # Effectuer une opération matricielle lourde
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    result = torch.matmul(x_gpu, x_gpu)
    end_time.record()
    
    # Attendre que l'opération soit terminée
    torch.cuda.synchronize()
    print(f"Opération matricielle effectuée sur GPU en {start_time.elapsed_time(end_time):.2f} ms")
    print("✅ GPU AMD correctement utilisé via DirectML")
except Exception as e:
    print(f"❌ Erreur lors du test GPU: {str(e)}")

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
with open("data/processed/train_camembert.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Extraction des labels uniques ===
all_slots = sorted({label for sample in raw_data for label in sample["slots"]})
all_intents = sorted({sample["intent"] for sample in raw_data})

slot2idx = {label: idx for idx, label in enumerate(all_slots)}
intent2idx = {label: idx for idx, label in enumerate(all_intents)}
idx2slot = {v: k for k, v in slot2idx.items()}
idx2intent = {v: k for k, v in intent2idx.items()}

# === Dataset complet
dataset = CamembertNLUDataset("data/processed/train_camembert.json", slot2idx, intent2idx)

# === Split train/valid
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# === Modèle
model = CamembertNLU(slot_size=len(slot2idx), intent_size=len(intent2idx))

# Forcer le modèle sur GPU
model.to(dml)
print("✅ Modèle déplacé sur GPU")

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
    output_dir="models/camembert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,   # Batch size minimal
    per_device_eval_batch_size=1,    # Batch size minimal
    num_train_epochs=4,
    weight_decay=0.1,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="slot_f1",
    save_total_limit=2,
    fp16=False,
    dataloader_num_workers=0,
)

# Patch pour forcer l'utilisation du GPU
class DirectMLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = dml
    
    def _prepare_inputs(self, inputs):
        """Place all inputs on the appropriate device"""
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

trainer = DirectMLTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Ajouter un gestionnaire d'erreurs pour capturer les problèmes potentiels
try:
    print("\n🚀 Début de l'entraînement avec DirectML sur GPU AMD...")
    print("💡 Vérifiez votre gestionnaire de tâches pour confirmer l'utilisation du GPU")
    trainer.train()
    print("✅ Entraînement terminé avec succès!")
except Exception as e:
    print(f"❌ Erreur pendant l'entraînement: {str(e)}")
    print("\n💡 Si le GPU n'est toujours pas utilisé, essayez cette alternative :")
    print("1. Installez PyTorch CPU uniquement : pip install torch")
    print("2. Utilisez Google Colab qui offre un GPU NVIDIA gratuit")
    raise

# === Sauvegarde
trainer.save_model("models/camembert_final")
print("✅ Modèle Camembert sauvegardé")

# Évaluation finale sur l'ensemble de validation
print("\n🔍 Évaluation finale du modèle:")
eval_results = trainer.evaluate()
print(f"Intent Accuracy: {eval_results['eval_intent_acc']:.4f}")
print(f"Slot F1-score: {eval_results['eval_slot_f1']:.4f}")
print(f"Slot Precision: {eval_results['eval_slot_precision']:.4f}")
print(f"Slot Recall: {eval_results['eval_slot_recall']:.4f}")

# Comparaison avec les objectifs du professeur
print("\n📊 Comparaison avec les objectifs:")
print(f"Intent Accuracy: {eval_results['eval_intent_acc']:.4f} (objectif: 0.7955)")
print(f"Slot Precision: {eval_results['eval_slot_precision']:.4f} (objectif: 0.5777)")
print(f"Slot Recall: {eval_results['eval_slot_recall']:.4f} (objectif: 0.6020)")
print(f"Slot F1-score: {eval_results['eval_slot_f1']:.4f} (objectif: 0.5899)")
