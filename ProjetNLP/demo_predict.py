import torch
from models.model_bilstm import BiLSTM_NLU
from utils import build_vocab, encode_sequences
import sys

# === Chargement du modèle sauvegardé ===
checkpoint = torch.load("ProjetNLP/models/bilstm_nlu.pt", map_location="cpu")

word2idx = checkpoint["word2idx"]
slot2idx = checkpoint["slot2idx"]
intent2idx = checkpoint["intent2idx"]

idx2slot = {v: k for k, v in slot2idx.items()}
idx2intent = {v: k for k, v in intent2idx.items()}

vocab_size = len(word2idx)
slot_size = len(slot2idx)
intent_size = len(intent2idx)
embedding_dim = 128
hidden_dim = 64
padding_idx = word2idx["<PAD>"]

# === Chargement du modèle ===
model = BiLSTM_NLU(vocab_size, embedding_dim, hidden_dim, slot_size, intent_size, padding_idx)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Fonction de prédiction ===
def predict(model, sentence):
    tokens = sentence.lower().split()
    input_ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        slot_logits, intent_logits = model(input_tensor)

        # Intent
        intent_idx = torch.argmax(intent_logits, dim=1).item()
        intent = idx2intent[intent_idx]

        # Slots
        slot_preds = torch.argmax(slot_logits, dim=2)[0].cpu().tolist()
        slot_labels = [idx2slot[p] for p in slot_preds]

    print("\n🧪 Résultat de la prédiction :")
    print("Phrase :", sentence)
    print("Intent :", intent)
    print("Slots  :")
    for token, slot in zip(tokens, slot_labels):
        print(f"  {token:<15} → {slot}")

# === Lancement interactif ===
if __name__ == "__main__":
    while True:
        try:
            sentence = input("\nTapez une phrase (ou 'exit' pour quitter) : ")
            if sentence.lower() in ["exit", "quit", "q", "stop", "fin", "stopper"]:
                break
            predict(model, sentence)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")