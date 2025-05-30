# NLP
Etudiant : Réadh Himeur
Structure : 


RenduNLP/
│
├── ProjetNLP/                             # Répertoire principal
│
├── dataset.py                         # Gestion des datasets (général)
├── dataset_camembert.py              # Dataset spécifique à CamemBERT
├── demo_predict.py                   # Script de prédiction
├── prepare_camembert_data.py        # Préparation des données pour CamemBERT
├── train_camembert.py               # Entraînement CamemBERT
├── figures/                          # Courbes d'analyse
│   ├── gap_comparison.png
│   └── performance_comparison.png
│   ├── data/                             # Données d'entraînement/validation/test
│   │   ├── train/
│   │   ├── dev/
│   │   ├── test/
│   │   └── processed/
│   └── models/                           # Modèles entraînés (pt, bin, safetensors)
│       ├── bilstm_nlu.pt
│       ├── model_bilstm.py
│       ├── model_camembert.py
│       ├── camembert/
│       │   └── checkpoint-20724/
│       └── camembert_final/
│
├── comparaison_model.py                  # Comparaison entre les modèles
├── demo_predict_bilstm.py                # Prédiction avec BiLSTM
├── demo_predict_camembert.py             # Prédiction avec CamemBERT
├── dataset.py                            # (copie à vérifier)
├── train_bilstm.py                       # Entraînement BiLSTM
├── train_camembert_amd.py                # Variante AMD pour CamemBERT
├── train_bert_colab.ipynb                # Notebook BERT sur Google Colab
├── logs/                                 # Logs TensorBoard
├── figures/                              # Graphiques comparatifs
├── utils.py                              # Fonctions utilitaires
├── .gitattributes                        # Configuration Git LFS
├── README.md
