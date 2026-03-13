"""
src/train.py
------------
Entraîne et évalue plusieurs modèles de scoring crédit sur Home Credit.
Sauvegarde le meilleur modèle + métriques au format JSON.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report, roc_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

sys.path.append(str(Path(__file__).parent))
from data_loader import run_pipeline

MODELS_DIR = Path("models")
PROC_DIR   = Path("data/processed")


# ─────────────────────────────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────────────────────────

def get_models() -> dict:
    models = {}

    models["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=1_000, C=0.1,
                                     class_weight="balanced", random_state=42)),
    ])

    models["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=20, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])

    if LGB_AVAILABLE:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, max_depth=-1,
            min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )

    if XGB_AVAILABLE:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=10,
            eval_metric="auc", use_label_encoder=False,
            random_state=42, n_jobs=-1, verbosity=0,
        )

    return models


# ─────────────────────────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate(model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # ROC curve (sous-échantillonné pour JSON léger)
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    step = max(1, len(fpr) // 200)

    metrics = {
        "name":           name,
        "accuracy":       float(accuracy_score(y_te, y_pred)),
        "roc_auc":        float(roc_auc_score(y_te, y_proba)),
        "avg_precision":  float(average_precision_score(y_te, y_proba)),
        "f1_score":       float(f1_score(y_te, y_pred)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
        "roc_curve": {
            "fpr": fpr[::step].tolist(),
            "tpr": tpr[::step].tolist(),
        },
    }
    return metrics


def get_importance(model, feature_names: list) -> dict:
    """Extrait l'importance des variables (top 30)."""
    try:
        if hasattr(model, "named_steps"):
            clf = model.named_steps["clf"]
        else:
            clf = model

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return {}

        d = dict(zip(feature_names, imp.tolist()))
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:30])
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────
# PIPELINE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────

def train_all(df_enc: pd.DataFrame, feature_names: list) -> tuple:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X = df_enc[feature_names]
    y = df_enc["TARGET"]

    # Sanitiser les noms de colonnes pour LightGBM (pas de caractères spéciaux JSON)
    import re
    def sanitize(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

    X.columns = [sanitize(c) for c in X.columns]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"📐 Split : {len(X_train):,} train / {len(X_test):,} test")
    print(f"   Défauts (train) : {y_train.mean()*100:.1f}%")

    # SMOTE uniquement si dataset < 50k (coûteux sinon)
    if len(X_train) < 50_000:
        print("⚖️  Application SMOTE…")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=feature_names)
        print(f"   Après SMOTE : {len(X_train):,} lignes")

    models    = get_models()
    all_metrics = {}
    best_auc  = 0.0
    best_name = ""

    print("\n🤖 Entraînement…")
    print("─" * 55)

    for name, model in models.items():
        print(f"\n▶  {name.replace('_', ' ').title()}")
        m = evaluate(model, X_train, X_test, y_train, y_test, name)
        m["feature_importance"] = get_importance(model, feature_names)
        all_metrics[name] = m

        print(f"   ROC-AUC : {m['roc_auc']:.4f}  |  Avg-Prec : {m['avg_precision']:.4f}  |  F1 : {m['f1_score']:.4f}")

        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

        if m["roc_auc"] > best_auc:
            best_auc  = m["roc_auc"]
            best_name = name

    # Sauvegardes
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    joblib.dump(joblib.load(MODELS_DIR / f"{best_name}.pkl"),
                MODELS_DIR / "best_model.pkl")

    (MODELS_DIR / "best_model_name.txt").write_text(best_name)

    print(f"\n🏆 Meilleur modèle : {best_name}  (AUC = {best_auc:.4f})")
    return all_metrics, best_name


# ─────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────

def run_training(sample_size: int = None):
    enc_path = PROC_DIR / "home_credit_encoded.csv"

    if not enc_path.exists():
        print("📥 Données non trouvées → pipeline data…")
        _, df_enc, feature_names = run_pipeline(sample_size=sample_size)
    else:
        print(f"✅ Données encodées chargées depuis {enc_path}")
        df_enc = pd.read_csv(enc_path)
        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                feature_names = json.load(f)
        else:
            feature_names = [c for c in df_enc.columns if c != "TARGET"]

    metrics, best = train_all(df_enc, feature_names)

    # Résumé
    print("\n" + "=" * 65)
    print(f"{'Modèle':<30} {'ROC-AUC':>10} {'Avg-Prec':>10} {'F1':>8}")
    print("─" * 65)
    for n, m in metrics.items():
        flag = " 🏆" if n == best else ""
        print(f"{n.replace('_',' ').title():<30} {m['roc_auc']:>10.4f} "
              f"{m['avg_precision']:>10.4f} {m['f1_score']:>8.4f}{flag}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Nombre de lignes (ex: 20000 pour un test rapide)")
    args = parser.parse_args()
    run_training(sample_size=args.sample)
