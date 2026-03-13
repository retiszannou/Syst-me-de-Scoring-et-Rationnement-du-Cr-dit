# Système de Scoring et Rationnement du Crédit

Un projet personnel que j'ai réalisé pour explorer le Machine Learning appliqué à la finance.
L'idée de base : à partir du profil d'un demandeur de crédit, est-ce qu'on peut prédire s'il va rembourser ou pas ?

J'ai utilisé le dataset **Home Credit Default Risk** disponible sur Kaggle (plus de 300 000 demandes de crédit réelles).

---

## C'est quoi le rationnement du crédit ?

En gros, c'est quand une banque refuse d'accorder un prêt, pas forcément parce que la personne est insolvable, mais parce qu'elle n'a pas assez d'informations pour évaluer le risque. C'est un concept théorisé par Stiglitz et Weiss en 1981.

Ce projet essaie de répondre à cette question avec le ML : **accorder ou rationner ?**

---

## Ce que fait l'application

- On entre le profil d'un demandeur (revenus, âge, historique de crédit, etc.)
- Le modèle calcule une probabilité de défaut
- Il retourne un score entre 300 et 850 et une décision : **ACCORDER** ou **RATIONNER**

---

## Technologies utilisées

- Python
- Streamlit pour l'interface
- LightGBM, XGBoost, Random Forest, Régression Logistique
- scikit-learn, pandas, numpy, plotly

---

## Lancer le projet

```bash
# Cloner le dépôt
git clone https://github.com/retiszannou/Syst-me-de-Scoring-et-Rationnement-du-Cr-dit.git
cd Syst-me-de-Scoring-et-Rationnement-du-Cr-dit

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle (sans Kaggle, des données synthétiques sont générées automatiquement)
python src/train.py

# Lancer l'app
streamlit run app.py
```

Si vous avez les données Kaggle, placez `application_train.csv` dans `data/raw/` avant de lancer `train.py`.

---

## Résultats obtenus

| Modèle | AUC-ROC |
|---|---|
| Logistic Regression | 0.71 |
| Random Forest | 0.73 |
| XGBoost | 0.75 |
| LightGBM | 0.76 |

LightGBM donne les meilleurs résultats sur ce dataset.

---

## Structure du projet

```
├── app.py               # Interface Streamlit
├── src/
│   ├── data_loader.py   # Chargement et préparation des données
│   ├── train.py         # Entraînement des modèles
│   └── predict.py       # Prédiction pour un profil donné
├── models/              # Modèles sauvegardés
├── tests/               # Tests unitaires
└── requirements.txt
```

---

## Référence principale

Stiglitz, J.E. & Weiss, A. (1981). *Credit Rationing in Markets with Imperfect Information*. American Economic Review.
