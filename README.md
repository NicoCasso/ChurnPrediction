# 🧠 TelcoNova - Prédiction de Churn Clients avec le Deep Learning

## 📌 Contexte

Ce projet a été réalisé dans le cadre d'une mission pour **TelcoNova**, opérateur télécom souhaitant **anticiper les départs de ses clients** (churn). En tant que Data Scientists au sein d’une ESN, notre objectif est de livrer un **prototype de modèle de prédiction** du churn, performant, reproductible et prêt pour l’intégration en production.

## 🎯 Objectifs

- Explorer, nettoyer et préparer les données CRM.
- Concevoir un pipeline de modélisation **Deep Learning** (MLP) sous **TensorFlow/Keras** ou **PyTorch**.
- Comparer les performances du modèle à une baseline simple.
- Fournir un modèle exportable avec ses artefacts pour une utilisation en production.
- Respecter les bonnes pratiques Git et assurer la reproductibilité.

## 🔧 Stack technique

- Python ≥ 3.9
- TensorFlow / Keras ou PyTorch
- scikit-learn, pandas, matplotlib, seaborn
- TensorBoard ou PyTorch Lightning pour le logging
- Imbalanced-learn (gestion du déséquilibre)
- KerasTuner / Optuna pour le tuning
- Git & GitHub

## 📂 Structure du projet

.
├── data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── img
│   ├── Evolution_of_training.png
│   ├── Matrice_de_Confusion.png
│   └── shap_summary_plot.png
├── model
│   └── model.pkl
├── notebooks
│   ├── CHURNPREDICTION
│   │   └── grid_search
│   │       ├── oracle.json
│   │       ├── trial_00
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_01
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_02
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_03
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_04
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_05
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_06
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_07
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_08
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       ├── trial_09
│   │       │   ├── build_config.json
│   │       │   ├── checkpoint.weights.h5
│   │       │   └── trial.json
│   │       └── tuner0.json
│   └── davidnotebook.ipynb
├── requirements.txt
└── src
    ├── deeplmodel.py
    ├── __init__.py
    ├── pipeline_functions.py
    └── regression_logistic.py

## ✅ Fonctionnalités livrées

### 🧼 Préparation des données

- Nettoyage et typage cohérents de toutes les colonnes
- Encodage des variables catégorielles (One-Hot ou Ordinal)
- Normalisation / standardisation des variables numériques
- Split stratifié : train / validation / test

### 🧠 Modélisation

- MLP (≥ 2 couches cachées) implémenté **from scratch**
- Fonction de perte : `BinaryCrossentropy`, avec pondération
- Optimiseur : `Adam`
- **Gestion du déséquilibre** : pondération ou sur-échantillonnage
- EarlyStopping + ModelCheckpoint
- Suivi via TensorBoard / LightningLogger

### 📊 Évaluation

- **Baseline** : régression logistique
- Métriques : **ROC-AUC**, **F1-score**, **Recall** (focus sur `Churn = Yes`)
- Visualisations : courbe ROC, matrice de confusion, courbes d’apprentissage
- Explication des variables influentes (feature importance + SHAP ou approches simples)

### 📦 Export / Inférence

- Modèle enregistré : `SavedModel` (TF) ou `.pt` (PyTorch)
- Encoders + scalers sauvegardés
- Script d’inférence prêt à l’emploi sur nouveaux clients

## 📌 Reproductibilité

- **Seed fixée** pour tous les composants pseudo-aléatoires
- Modèle sauvegardé et restaurable
- Instructions de réexécution documentées
- Suivi de toutes les fonctionnalités via branches Git + pull requests

## 🔍 Lancer le projet

### Installation

```bash
git clone https://github.com/<votre-org>/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

| Modèle            | ROC-AUC  | F1-Score | Recall (Churn) |
| ----------------- | -------- | -------- | -------------- |
| Baseline (LogReg) | 0.76     | 0.58     | 0.61           |
| MLP Final         | **0.83** | **0.63** | **0.68**       |

## 📚 Ressources

- [TensorFlow API Docs](https://www.tensorflow.org/api_docs)
- [PyTorch Docs](https://pytorch.org/docs/)
- [KerasTuner](https://keras.io/keras_tuner/)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)
- *DeepLearning.AI TensorFlow Developer* (Coursera)
- *PyTorch Lightning Crash Course* (YouTube)

## 👥 Auteurs

- **David** (Data Scientist)
- **Nicolas** (Data Scientist)
