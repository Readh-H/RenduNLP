#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de comparaison des performances des modèles BiLSTM et CamemBERT pour NLU
Auteur: Manus
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import pandas as pd
from tabulate import tabulate

# Définir les résultats des modèles
# Ces valeurs sont basées sur les résultats obtenus précédemment
bilstm_results = {
    "intent_accuracy": 0.7645,
    "slot_precision": 0.5672,
    "slot_recall": 0.5190,
    "slot_f1": 0.5420,
    "training_time": 300  # en secondes (approximatif)
}

camembert_results = {
    "intent_accuracy": 0.8576,
    "slot_precision": 0.6138,
    "slot_recall": 0.6640,
    "slot_f1": 0.6379,
    "training_time": 7200  # en secondes (approximatif)
}

# Objectifs fixés par le professeur
objectives = {
    "intent_accuracy": 0.7955,
    "slot_precision": 0.5777,
    "slot_recall": 0.6020,
    "slot_f1": 0.5899
}

# Créer le dossier pour les graphiques s'il n'existe pas
os.makedirs("figures", exist_ok=True)

# Fonction pour calculer l'écart par rapport aux objectifs
def calculate_gap(value, objective):
    return ((value - objective) / objective) * 100

# Calculer les écarts
bilstm_gaps = {
    metric: calculate_gap(bilstm_results[metric], objective)
    for metric, objective in objectives.items()
}

camembert_gaps = {
    metric: calculate_gap(camembert_results[metric], objective)
    for metric, objective in objectives.items()
}

# 1. Graphique de comparaison des performances
def plot_performance_comparison():
    metrics = ["intent_accuracy", "slot_precision", "slot_recall", "slot_f1"]
    metric_names = ["Intent Accuracy", "Slot Precision", "Slot Recall", "Slot F1-score"]
    
    bilstm_values = [bilstm_results[m] for m in metrics]
    camembert_values = [camembert_results[m] for m in metrics]
    objective_values = [objectives[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, bilstm_values, width, label='BiLSTM', color='#3498db')
    rects2 = ax.bar(x, camembert_values, width, label='CamemBERT', color='#e74c3c')
    rects3 = ax.bar(x + width, objective_values, width, label='Objectifs', color='#2ecc71')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des performances des modèles', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=300)
    plt.close()
    
    print("✅ Graphique de comparaison des performances généré : figures/performance_comparison.png")

# 2. Graphique des écarts par rapport aux objectifs
def plot_gap_comparison():
    metrics = ["intent_accuracy", "slot_precision", "slot_recall", "slot_f1"]
    metric_names = ["Intent Accuracy", "Slot Precision", "Slot Recall", "Slot F1-score"]
    
    bilstm_gap_values = [bilstm_gaps[m] for m in metrics]
    camembert_gap_values = [camembert_gaps[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, bilstm_gap_values, width, label='BiLSTM', color='#3498db')
    rects2 = ax.bar(x + width/2, camembert_gap_values, width, label='CamemBERT', color='#e74c3c')
    
    # Ajouter une ligne horizontale à 0%
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_ylabel('Écart par rapport aux objectifs (%)', fontsize=12)
    ax.set_title('Écart des performances par rapport aux objectifs', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('figures/gap_comparison.png', dpi=300)
    plt.close()
    
    print("✅ Graphique des écarts par rapport aux objectifs généré : figures/gap_comparison.png")

# 3. Graphique radar des performances
def plot_radar_chart():
    metrics = ["Intent Accuracy", "Slot Precision", "Slot Recall", "Slot F1-score"]
    
    # Convertir les métriques en format radar (répéter le premier point)
    values_bilstm = [
        bilstm_results["intent_accuracy"],
        bilstm_results["slot_precision"],
        bilstm_results["slot_recall"],
        bilstm_results["slot_f1"]
    ]
    values_bilstm = values_bilstm + [values_bilstm[0]]
    
    values_camembert = [
        camembert_results["intent_accuracy"],
        camembert_results["slot_precision"],
        camembert_results["slot_recall"],
        camembert_results["slot_f1"]
    ]
    values_camembert = values_camembert + [values_camembert[0]]
    
    values_objectives = [
        objectives["intent_accuracy"],
        objectives["slot_precision"],
        objectives["slot_recall"],
        objectives["slot_f1"]
    ]
    values_objectives = values_objectives + [values_objectives[0]]
    
    # Répéter les labels pour fermer le polygone
    metrics = metrics + [metrics[0]]
    
    # Calculer l'angle pour chaque métrique
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values_bilstm, 'o-', linewidth=2, label='BiLSTM', color='#3498db')
    ax.fill(angles, values_bilstm, alpha=0.1, color='#3498db')
    
    ax.plot(angles, values_camembert, 'o-', linewidth=2, label='CamemBERT', color='#e74c3c')
    ax.fill(angles, values_camembert, alpha=0.1, color='#e74c3c')
    
    ax.plot(angles, values_objectives, 'o-', linewidth=2, label='Objectifs', color='#2ecc71')
    ax.fill(angles, values_objectives, alpha=0.1, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics[:-1], fontsize=12)
    
    # Définir les limites du graphique
    ax.set_ylim(0, 1)
    
    # Ajouter une grille
    ax.grid(True)
    
    # Ajouter une légende
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title('Comparaison des performances sur un graphique radar', fontsize=16, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig('figures/radar_chart.png', dpi=300)
    plt.close()
    
    print("✅ Graphique radar généré : figures/radar_chart.png")

# 4. Graphique de comparaison des temps d'entraînement
def plot_training_time():
    models = ['BiLSTM', 'CamemBERT']
    times = [bilstm_results["training_time"] / 60, camembert_results["training_time"] / 60]  # Conversion en minutes
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=['#3498db', '#e74c3c'])
    
    ax.set_ylabel('Temps d\'entraînement (minutes)', fontsize=12)
    ax.set_title('Comparaison des temps d\'entraînement', fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} min',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/training_time.png', dpi=300)
    plt.close()
    
    print("✅ Graphique des temps d'entraînement généré : figures/training_time.png")

# 5. Tableau de comparaison détaillé
def generate_comparison_table():
    data = []
    
    metrics = [
        ("Intent Accuracy", "intent_accuracy"),
        ("Slot Precision", "slot_precision"),
        ("Slot Recall", "slot_recall"),
        ("Slot F1-score", "slot_f1"),
        ("Temps d'entraînement", "training_time")
    ]
    
    for display_name, metric_key in metrics:
        if metric_key == "training_time":
            bilstm_value = f"{bilstm_results[metric_key] / 60:.1f} min"
            camembert_value = f"{camembert_results[metric_key] / 60:.1f} min"
            objective = "N/A"
            bilstm_gap = "N/A"
            camembert_gap = "N/A"
        else:
            bilstm_value = f"{bilstm_results[metric_key]:.4f}"
            camembert_value = f"{camembert_results[metric_key]:.4f}"
            objective = f"{objectives[metric_key]:.4f}"
            bilstm_gap = f"{bilstm_gaps[metric_key]:.2f}%"
            camembert_gap = f"{camembert_gaps[metric_key]:.2f}%"
        
        data.append([
            display_name,
            bilstm_value,
            camembert_value,
            objective,
            bilstm_gap,
            camembert_gap
        ])
    
    # Créer un DataFrame pandas
    df = pd.DataFrame(data, columns=[
        "Métrique",
        "BiLSTM",
        "CamemBERT",
        "Objectif",
        "Écart BiLSTM",
        "Écart CamemBERT"
    ])
    
    # Sauvegarder en CSV
    df.to_csv("figures/comparison_table.csv", index=False)
    
    # Sauvegarder en format Markdown
    with open("figures/comparison_table.md", "w", encoding="utf-8") as f:
        f.write(tabulate(data, headers=[
            "Métrique",
            "BiLSTM",
            "CamemBERT",
            "Objectif",
            "Écart BiLSTM",
            "Écart CamemBERT"
        ], tablefmt="pipe"))
    
    # Sauvegarder en format HTML
    with open("figures/comparison_table.html", "w", encoding="utf-8") as f:
        f.write(df.to_html(index=False))
    
    print("✅ Tableau de comparaison généré : figures/comparison_table.csv, .md, .html")
    
    # Afficher le tableau dans la console
    print("\n" + "="*80)
    print("TABLEAU DE COMPARAISON DES PERFORMANCES")
    print("="*80)
    print(tabulate(data, headers=[
        "Métrique",
        "BiLSTM",
        "CamemBERT",
        "Objectif",
        "Écart BiLSTM",
        "Écart CamemBERT"
    ], tablefmt="grid"))
    print("="*80)

# 6. Sauvegarder les résultats en JSON pour une utilisation ultérieure
def save_results_json():
    results = {
        "bilstm": bilstm_results,
        "camembert": camembert_results,
        "objectives": objectives,
        "bilstm_gaps": bilstm_gaps,
        "camembert_gaps": camembert_gaps
    }
    
    with open("figures/model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Résultats sauvegardés en JSON : figures/model_comparison_results.json")

# 7. Générer un rapport de synthèse
def generate_summary_report():
    # Calculer les améliorations de CamemBERT par rapport à BiLSTM
    improvements = {
        metric: ((camembert_results[metric] - bilstm_results[metric]) / bilstm_results[metric]) * 100
        for metric in ["intent_accuracy", "slot_precision", "slot_recall", "slot_f1"]
    }
    
    report = f"""# Rapport de comparaison des modèles BiLSTM et CamemBERT

## Résumé des performances

Ce rapport compare les performances des modèles BiLSTM et CamemBERT pour les tâches de Natural Language Understanding (NLU).

### Performances globales

- **BiLSTM** : Modèle classique de Deep Learning, plus léger et plus rapide à entraîner
- **CamemBERT** : Modèle Transformer pré-entraîné, plus performant mais plus lourd

### Comparaison par rapport aux objectifs

| Métrique | BiLSTM | CamemBERT | Objectif | Écart BiLSTM | Écart CamemBERT |
|----------|--------|-----------|----------|--------------|-----------------|
| Intent Accuracy | {bilstm_results["intent_accuracy"]:.4f} | {camembert_results["intent_accuracy"]:.4f} | {objectives["intent_accuracy"]:.4f} | {bilstm_gaps["intent_accuracy"]:.2f}% | {camembert_gaps["intent_accuracy"]:.2f}% |
| Slot Precision | {bilstm_results["slot_precision"]:.4f} | {camembert_results["slot_precision"]:.4f} | {objectives["slot_precision"]:.4f} | {bilstm_gaps["slot_precision"]:.2f}% | {camembert_gaps["slot_precision"]:.2f}% |
| Slot Recall | {bilstm_results["slot_recall"]:.4f} | {camembert_results["slot_recall"]:.4f} | {objectives["slot_recall"]:.4f} | {bilstm_gaps["slot_recall"]:.2f}% | {camembert_gaps["slot_recall"]:.2f}% |
| Slot F1-score | {bilstm_results["slot_f1"]:.4f} | {camembert_results["slot_f1"]:.4f} | {objectives["slot_f1"]:.4f} | {bilstm_gaps["slot_f1"]:.2f}% | {camembert_gaps["slot_f1"]:.2f}% |

### Amélioration apportée par CamemBERT

- Intent Accuracy : **+{improvements["intent_accuracy"]:.2f}%**
- Slot Precision : **+{improvements["slot_precision"]:.2f}%**
- Slot Recall : **+{improvements["slot_recall"]:.2f}%**
- Slot F1-score : **+{improvements["slot_f1"]:.2f}%**

### Temps d'entraînement

- BiLSTM : **{bilstm_results["training_time"] / 60:.1f} minutes**
- CamemBERT : **{camembert_results["training_time"] / 60:.1f} minutes** ({camembert_results["training_time"] / bilstm_results["training_time"]:.1f}x plus long)

## Conclusion

Le modèle CamemBERT surpasse significativement le modèle BiLSTM sur toutes les métriques d'évaluation, avec des améliorations allant de {min(improvements.values()):.1f}% à {max(improvements.values()):.1f}%. Cette supériorité s'explique par la puissance des architectures Transformer et l'avantage du pré-entraînement sur de larges corpus français.

Cependant, cette amélioration des performances s'accompagne d'un coût computationnel plus élevé, avec un temps d'entraînement {camembert_results["training_time"] / bilstm_results["training_time"]:.1f} fois plus long pour CamemBERT.

Le choix entre ces deux modèles dépendra donc des contraintes spécifiques du projet :
- Pour des applications nécessitant une haute précision : **CamemBERT**
- Pour des applications avec des ressources limitées ou des contraintes de temps : **BiLSTM**

![Comparaison des performances](performance_comparison.png)
![Écarts par rapport aux objectifs](gap_comparison.png)
![Graphique radar](radar_chart.png)
![Temps d'entraînement](training_time.png)
"""
    
    with open("figures/model_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Rapport de synthèse généré : figures/model_comparison_report.md")

# Fonction principale
def main():
    print("\n" + "="*50)
    print("🚀 Comparaison des modèles BiLSTM et CamemBERT")
    print("="*50)
    
    # Générer tous les graphiques et tableaux
    plot_performance_comparison()
    plot_gap_comparison()
    plot_radar_chart()
    plot_training_time()
    generate_comparison_table()
    save_results_json()
    generate_summary_report()
    
    print("\n✅ Tous les graphiques et tableaux ont été générés dans le dossier 'figures/'")
    print("📊 Vous pouvez maintenant utiliser ces visualisations dans votre rapport")

if __name__ == "__main__":
    main()
