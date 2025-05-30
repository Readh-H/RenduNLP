# NLP
Etudiant : Réadh Himeur
Structure : 
Attention : Ci-dessous les fichiers à prendre en compte
```
RenduNLP/
│
├── ProjetNLP/                            # Répertoire principal
├── demo_predict_bilstm.py                # A UTILISER Prédiction avec BiLSTM
├── demo_predict_camembert.py             # A UTILISER Prédiction avec CamemBERT
├── train_camembert.py                    # Entraînement CamemBERT
├── train_bilstm.py                       # Entraînement BiLSTM
├── dataset.py                            # Gestion des datasets (général)
├── dataset_camembert.py                  # Dataset spécifique à CamemBERT
├── demo_predict.py                       # Script de prédiction
├── prepare_camembert_data.py             # Préparation des données pour CamemBERT
│
└── models/                               # Modèles entraînés (pt, bin, safetensors)
│    ├── bilstm_nlu.pt                    #Fichier de sauvagarde modèle BILSTM
│    └── camembert_final/                 #Dossier de sauvagarde modèle CAMEMBERT
├── figures/                              # histo
│   ├── gap_comparison.png
│   └── performance_comparison.png
├── data/                                 # Données d'entraînement/validation/test
│   ├── train/
│   ├── dev/
│   ├── test/
│   └── processed/
│
└── README.md
