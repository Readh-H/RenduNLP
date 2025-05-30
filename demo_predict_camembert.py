import torch
from transformers import CamembertTokenizerFast
import json
from models.model_camembert import CamembertNLU
import sys

# === Chargement des mappings ===
try:
    with open("data/processed/train_camembert.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_slots = sorted({label for sample in raw_data for label in sample["slots"]})
    all_intents = sorted({sample["intent"] for sample in raw_data})

    slot2idx = {label: idx for idx, label in enumerate(all_slots)}
    intent2idx = {label: idx for idx, label in enumerate(all_intents)}
    idx2slot = {v: k for k, v in slot2idx.items()}
    idx2intent = {v: k for k, v in intent2idx.items()}
except Exception as e:
    print(f"❌ Erreur lors du chargement des mappings: {e}")
    print("⚠️ Vérifiez que le fichier train_camembert.json existe dans le dossier data/processed/")
    sys.exit(1)

# === Chargement du modèle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CamembertNLU(slot_size=len(slot2idx), intent_size=len(intent2idx))

# Essayer différentes méthodes de chargement
try:
    # 1. Essayer safetensors (format recommandé)
    try:
        from safetensors.torch import load_file
        state_dict = load_file("models/camembert_final/model.safetensors")
        model.load_state_dict(state_dict)
        print("✅ Modèle chargé depuis safetensors avec succès!")
    except Exception as e:
        print(f"ℹ️ Impossible de charger depuis safetensors: {e}")
        raise Exception("Essayer méthode suivante")
        
except Exception:
    # 2. Essayer pytorch_model.bin
    try:
        model.load_state_dict(torch.load("models/camembert_final/pytorch_model.bin", map_location=device))
        print("✅ Modèle chargé depuis pytorch_model.bin avec succès!")
    except Exception as e:
        print(f"ℹ️ Impossible de charger depuis pytorch_model.bin: {e}")
        
        # 3. Essayer from_pretrained
        try:
            from transformers import CamembertForTokenClassification
            print("Chargement du modèle avec from_pretrained...")
            encoder = CamembertForTokenClassification.from_pretrained("models/camembert_final/").roberta
            model.encoder = encoder
            print("✅ Modèle chargé avec from_pretrained avec succès!")
        except Exception as e2:
            print(f"❌ Toutes les méthodes de chargement ont échoué: {e2}")
            print("⚠️ Veuillez réentraîner le modèle et sauvegarder explicitement les poids.")
            sys.exit(1)

model.to(device)
model.eval()

# === Tokenizer ===
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

# === Fonction de prédiction ===
def predict(model, sentence):
    tokens = sentence.split()
    
    # Tokenisation avec is_split_into_words=True
    # IMPORTANT: Garder l'objet encodé original pour word_ids
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True)
    
    # Récupérer les word_ids avant de convertir en tenseurs pour le GPU <-- echec
    word_ids = encoded.word_ids(batch_index=0)
    
    # Préparer les entrées pour le modèle
    inputs = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    slot_logits, intent_logits = outputs
    
    # Intent
    intent_pred = torch.argmax(intent_logits, dim=1).item()
    intent = idx2intent[intent_pred]
    
    # Slots
    slot_preds = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    
    # Aligner les prédictions avec les tokens
    previous_word_idx = None
    slot_labels = []
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        slot_labels.append(idx2slot[slot_preds[i]])
        previous_word_idx = word_idx
    
    print("\n🧪 Résultat de la prédiction :")
    print("Phrase :", sentence)
    print("Intent :", intent)
    print("Slots  :")
    for token, slot in zip(tokens, slot_labels):
        print(f"  {token:<15} → {slot}")

# === Lancement interactif ===
if __name__ == "__main__":
    print("🤖 Démo du modèle CamemBERT NLU")
    print("💡 Tapez une phrase pour obtenir l'intention et les slots détectés")
    print(f"📊 Modèle chargé sur {device}")
    
    while True:
        try:
            sentence = input("\nTapez une phrase (ou 'exit' pour quitter) : ")
            if sentence.lower() in ["exit", "quit", "q", "stop", "fin", "stopper"]:
                break
            predict(model, sentence)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            print(f"Détails : {type(e).__name__}")
