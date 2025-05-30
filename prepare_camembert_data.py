import os
import pandas as pd

# === Fichiers d'entrée ===
IN_PATH = "ProjetNLP/data/train/slot-filling.in"
OUT_PATH = "ProjetNLP/data/train/slot-filling.out"
INTENT_PATH = "ProjetNLP/data/train/intentlabels"

# === Lecture des fichiers ===
with open(IN_PATH, encoding='utf-8') as f:
    sentences = [line.strip() for line in f.readlines()]

with open(OUT_PATH, encoding='utf-8') as f:
    slots = [line.strip() for line in f.readlines()]

with open(INTENT_PATH, encoding='utf-8') as f:
    intents = [line.strip() for line in f.readlines()]

# === Vérification ===
assert len(sentences) == len(slots) == len(intents), "Fichiers désalignés !"

# === Préparation des données ===
data = []
for sentence, slot_line, intent in zip(sentences, slots, intents):
    tokens = sentence.split()
    slot_tags = slot_line.split()

    if len(tokens) != len(slot_tags):
        # Skip lignes incohérentes
        continue

    data.append({
        "tokens": tokens,
        "slots": slot_tags,
        "intent": intent
    })

df = pd.DataFrame(data)
print(f"✅ Données chargées : {len(df)} phrases")

# === Sauvegarde CSV ===
os.makedirs("ProjetNLP/data/processed", exist_ok=True)
df.to_json("ProjetNLP/data/processed/train_camembert.json", orient="records", indent=2, force_ascii=False)

print("📁 Données sauvegardées dans : ProjetNLP/data/processed/train_camembert.json")
