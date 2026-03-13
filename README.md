# 🏦 CréditScope — Scoring & Rationnement du Crédit

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-green.svg)](https://lightgbm.readthedocs.io)

> Application de scoring crédit par Machine Learning pour décider d'**accorder ou rationner** un crédit selon le profil du demandeur.
> Basé sur le **Home Credit Default Risk Dataset** (Kaggle — 307 511 demandes, 122 variables).

---

## 🎬 Démo

| Évaluation d'un profil | Exploration des données | Performance des modèles |
|:---:|:---:|:---:|
| Interface de saisie + gauge score | Distributions & corrélations | Courbes ROC, matrice de confusion |

---

## 🎯 Objectif

Le **rationnement du crédit** (Stiglitz & Weiss, 1981) est le refus d'accorder un prêt malgré la volonté de payer le taux en vigueur — phénomène dû à l'asymétrie d'information.

Ce projet implémente un système de scoring qui :
1. Calcule un **score crédit [300–850]** à partir du profil d'un demandeur
2. Retourne une décision **ACCORDER / RATIONNER** avec probabilité de défaut
3. Explique les facteurs ayant conduit à la décision

---

## 🚀 Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/credit-scoring-home-credit.git
cd credit-scoring-home-credit

# 2. Environnement virtuel
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Préparer données + entraîner
#    Option A — avec vos fichiers Kaggle dans data/raw/
python src/train.py

#    Option B — sans Kaggle (génère ~5 000 observations synthétiques)
python src/train.py --sample 5000

# 5. Lancer l'app
streamlit run app.py
```

L'application est accessible sur **http://localhost:8501**

---

## 📦 Données — Home Credit Default Risk

Télécharger depuis [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) et placer dans `data/raw/` :

```
data/raw/
├── application_train.csv    ← obligatoire
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── credit_card_balance.csv
└── POS_CASH_balance.csv
```

> **Sans Kaggle ?** Le pipeline génère automatiquement des données synthétiques réalistes.

---

## 📁 Structure

```
credit-scoring/
├── app.py                    # Application Streamlit (4 pages)
├── src/
│   ├── data_loader.py        # Acquisition, nettoyage, feature engineering
│   ├── train.py              # Entraînement multi-modèles + sauvegarde
│   └── predict.py            # Scoring d'un profil individuel
├── data/
│   ├── raw/                  # Fichiers Kaggle bruts
│   └── processed/            # CSV nettoyés & encodés
├── models/                   # Modèles .pkl + metrics.json
├── tests/                    # Tests unitaires pytest
├── .streamlit/config.toml    # Thème sombre
├── .github/workflows/ci.yml  # Pipeline CI GitHub Actions
└── requirements.txt
```

---

## 🤖 Modèles

| Modèle | AUC-ROC* | Avg Precision |
|---|---|---|
| Logistic Regression | ~0.71 | ~0.24 |
| Random Forest | ~0.73 | ~0.27 |
| **LightGBM** | **~0.76** | **~0.31** |
| XGBoost | ~0.75 | ~0.30 |

\* Sur données synthétiques — les scores sur données Kaggle réelles sont significativement meilleurs.

### Features construites

- `CREDIT_INCOME_RATIO` — ratio crédit / revenu annuel
- `ANNUITY_INCOME_RATIO` — ratio mensualité / revenu
- `AGE_YEARS`, `EMPLOYED_YEARS`, `EMPLOYED_RATIO`
- `EXT_SOURCES_MEAN / MIN / STD` — agrégation des 3 scores externes
- `SOCIAL_CIRCLE_DEF_RATE` — taux de défaut dans l'entourage

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## ☁️ Déploiement Streamlit Cloud

1. Pusher sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le dépôt → `app.py` → **Deploy**

---

## 📚 Références

- Stiglitz, J.E. & Weiss, A. (1981). *Credit Rationing in Markets with Imperfect Information*. American Economic Review.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Home Credit Group (2018). [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

---

## ⚠️ Avertissement

Projet **éducatif**. Les décisions réelles nécessitent le respect du RGPD, des réglementations bancaires et des principes d'équité algorithmique.

---

## 📄 Licence

[MIT](LICENSE) © 2024
