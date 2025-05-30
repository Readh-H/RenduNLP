### Chargement des données NLU
def load_nlu_data_from_files(text_file, slot_file, intent_file):
    """
    Charge les données NLU à partir de fichiers contenant :
    - une phrase par ligne dans text_file (tokens séparés par des espaces)
    - les tags BIO correspondants dans slot_file (même structure)
    - une intention par ligne dans intent_file

    Returns:
        sentences (List[List[str]])
        slots (List[List[str]])
        intents (List[str])
    """
    with open(text_file, encoding='utf-8') as f:
        sentences = [line.strip().split() for line in f if line.strip()]

    with open(slot_file, encoding='utf-8') as f:
        slots = [line.strip().split() for line in f if line.strip()]

    with open(intent_file, encoding='utf-8') as f:
        intents = [line.strip() for line in f if line.strip()]

    print(f"Nombre de phrases   : {len(sentences)}")
    print(f"Nombre de slots     : {len(slots)}")
    print(f"Nombre d'intentions : {len(intents)}")

    # Nettoyage :suppression des entêtes parasites
    if sentences[0][0].lower() == "word" and slots[0][0].lower() == "slot":
        print("🧹 Suppression de l'entête parasite...")
        sentences = sentences[1:]
        slots = slots[1:]
        intents = intents[1:]


    for i in range(min(len(sentences), len(slots))):
        if len(sentences[i]) != len(slots[i]):
            print(f"🔴 Problème alignement tokens/slots à la phrase {i}")
            print("Phrase :", sentences[i])
            print("Slots  :", slots[i])
            break

    assert len(sentences) == len(slots) == len(intents), "Incohérence entre les fichiers"
    return sentences, slots, intents

### Vocabulaire et encodage

from collections import defaultdict

def build_vocab(sequences, add_pad_unk=True):
    """
    Construit un vocabulaire à partir d'une liste de séquences de tokens.
    Args:
        sequences: liste de listes de tokens (phrases ou étiquettes)
        add_pad_unk: si True, ajoute <PAD> et <UNK> au vocabulaire
    Returns:
        dict token -> index
    """
    vocab = {}
    index = 0

    if add_pad_unk:
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        index = 2
    else:
        index = 0

    for seq in sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

def encode_sequences(sequences, vocab, unk_token="<UNK>"):
    """
    Encode des séquences de tokens en entiers.
    """
    unk_index = vocab.get(unk_token, 1)
    return [[vocab.get(token, unk_index) for token in seq] for seq in sequences]

def encode_labels(sequences, vocab):
    """
    Encode des séquences d'étiquettes (slots) en entiers.
    """
    return [[vocab[label] for label in seq] for seq in sequences]
