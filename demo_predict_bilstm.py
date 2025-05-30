import torch
import sys
import os

# Vérifier si le dossier models existe
if not os.path.exists("models"):
    os.makedirs("models")

# Définition de la classe BiLSTM_NLU directement dans le script pour éviter les problèmes d'importation
class BiLSTM_NLU(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, slot_size, intent_size, padding_idx=0):
        super(BiLSTM_NLU, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Utilisation de "lstm" comme nom d'attribut pour être cohérent avec le modèle sauvegardé
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Couche pour la classification des slots
        self.slot_classifier = torch.nn.Linear(hidden_dim * 2, slot_size)
        
        # Couche pour la classification des intentions
        self.intent_classifier = torch.nn.Linear(hidden_dim * 2, intent_size)
    
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

# === Chargement du modèle sauvegardé ===
try:
    print("🔍 Chargement du modèle depuis models/bilstm_nlu.pt...")
    checkpoint = torch.load("models/bilstm_nlu.pt", map_location="cpu")
    print("✅ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    print("⚠️ Assurez-vous que le fichier models/bilstm_nlu.pt existe.")
    sys.exit(1)

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
try:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Poids du modèle chargés avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement des poids: {e}")
    sys.exit(1)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Utilisation du périphérique: {device}")
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
        slot_labels = [idx2slot[p] for p in slot_preds[:len(tokens)]]  # Limiter aux tokens d'entrée

    print("\n🧪 Résultat de la prédiction :")
    print("Phrase :", sentence)
    print("Intent :", intent)
    print("Slots  :")
    for token, slot in zip(tokens, slot_labels):
        print(f"  {token:<15} → {slot}")

# === Lancement interactif ===
if __name__ == "__main__":
    print("🤖 Démo du modèle BiLSTM NLU")
    print("💡 Tapez une phrase pour obtenir l'intention et les slots détectés")
    
    while True:
        try:
            sentence = input("\nTapez une phrase (ou 'exit' pour quitter) : ")
            if sentence.lower() in ["exit", "quit", "q", "stop", "fin", "stopper"]:
                break
            predict(model, sentence)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            print(f"Type d'erreur : {type(e).__name__}")
