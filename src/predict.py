"""
src/predict.py
--------------
Scoring d'un profil individuel : probabilité de défaut + décision.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

MODELS_DIR = Path("models")


# ─────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────

def load_model(model_name: str = "best"):
    path = MODELS_DIR / ("best_model.pkl" if model_name == "best" else f"{model_name}.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {path}. Lancez python src/train.py")
    return joblib.load(path)


def load_feature_names() -> list:
    path = MODELS_DIR / "feature_names.json"
    if not path.exists():
        raise FileNotFoundError("feature_names.json manquant. Lancez python src/train.py")
    return json.loads(path.read_text())


# ─────────────────────────────────────────────────────────────────
# PRÉPARATION DES FEATURES
# ─────────────────────────────────────────────────────────────────

def profile_to_dataframe(profile: Dict) -> pd.DataFrame:
    """Convertit le dict profil en DataFrame avec feature engineering."""
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_loader import engineer_features, encode_categoricals, clean_and_impute

    df = pd.DataFrame([profile])
    df = clean_and_impute(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df


def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Aligne le vecteur de features avec celui vu à l'entraînement."""
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ─────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────

def probability_to_score(prob_default: float) -> int:
    """
    Convertit P(défaut) en score crédit [300, 850].
    Score élevé = faible risque (convention FICO-like).
    """
    return int(300 + (1 - prob_default) * 550)


def score_to_risk_band(score: int) -> Tuple[str, str]:
    """Retourne le niveau de risque et la couleur associée."""
    if score >= 740:
        return "FAIBLE", "#2ecc71"
    elif score >= 670:
        return "MODÉRÉ", "#f39c12"
    elif score >= 580:
        return "ÉLEVÉ", "#e67e22"
    else:
        return "TRÈS ÉLEVÉ", "#e74c3c"


def predict(profile: Dict, model_name: str = "best") -> Dict:
    """
    Prédit le risque crédit pour un profil donné.

    Returns
    -------
    dict avec clés :
        decision       : "ACCORDER" | "RATIONNER"
        prob_default   : probabilité de défaut [0, 1]
        prob_repay     : probabilité de remboursement [0, 1]
        score          : score crédit [300, 850]
        risk_band      : "FAIBLE" | "MODÉRÉ" | "ÉLEVÉ" | "TRÈS ÉLEVÉ"
        risk_color     : code hex
        df_aligned     : DataFrame aligné (pour explainability)
    """
    model         = load_model(model_name)
    feature_names = load_feature_names()

    df         = profile_to_dataframe(profile)
    df_aligned = align_features(df, feature_names)

    prob_default = float(model.predict_proba(df_aligned)[0][1])
    prob_repay   = 1 - prob_default
    prediction   = int(model.predict(df_aligned)[0])

    decision = "ACCORDER" if prediction == 0 else "RATIONNER"
    score    = probability_to_score(prob_default)
    risk_band, risk_color = score_to_risk_band(score)

    return {
        "decision":     decision,
        "prob_default": prob_default,
        "prob_repay":   prob_repay,
        "score":        score,
        "risk_band":    risk_band,
        "risk_color":   risk_color,
        "df_aligned":   df_aligned,
    }


# ─────────────────────────────────────────────────────────────────
# EXEMPLE
# ─────────────────────────────────────────────────────────────────

EXAMPLE_PROFILE = {
    "AMT_CREDIT":          202_500,
    "AMT_INCOME_TOTAL":    67_500,
    "AMT_ANNUITY":         7_956,
    "AMT_GOODS_PRICE":     180_000,
    "DAYS_BIRTH":          -12_000,   # ~33 ans
    "DAYS_EMPLOYED":       -2_000,    # ~5.5 ans
    "DAYS_REGISTRATION":   -5_000,
    "DAYS_ID_PUBLISH":     -1_500,
    "CNT_FAM_MEMBERS":     2,
    "CNT_CHILDREN":        0,
    "EXT_SOURCE_1":        0.60,
    "EXT_SOURCE_2":        0.72,
    "EXT_SOURCE_3":        0.55,
    "REGION_POPULATION_RELATIVE": 0.035,
    "HOUR_APPR_PROCESS_START":    10,
    "OBS_30_CNT_SOCIAL_CIRCLE":   2,
    "DEF_30_CNT_SOCIAL_CIRCLE":   0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    "NAME_CONTRACT_TYPE":    "Cash loans",
    "CODE_GENDER":           "F",
    "FLAG_OWN_CAR":          "N",
    "FLAG_OWN_REALTY":       "Y",
    "NAME_TYPE_SUITE":       "Unaccompanied",
    "NAME_INCOME_TYPE":      "Working",
    "NAME_EDUCATION_TYPE":   "Higher education",
    "NAME_FAMILY_STATUS":    "Married",
    "NAME_HOUSING_TYPE":     "House / apartment",
    "OCCUPATION_TYPE":       "Managers",
    "ORGANIZATION_TYPE":     "Business Entity Type 3",
    "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
    "REGION_RATING_CLIENT":  2,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "FLAG_EMAIL":             1,
    "FLAG_PHONE":             1,
    "FLAG_WORK_PHONE":        0,
    "FLAG_DOCUMENT_3":        1,
    "LIVE_CITY_NOT_WORK_CITY": 0,
}

if __name__ == "__main__":
    try:
        result = predict(EXAMPLE_PROFILE)
        print(f"Décision       : {result['decision']}")
        print(f"P(défaut)      : {result['prob_default']:.3f}")
        print(f"P(remboursement): {result['prob_repay']:.3f}")
        print(f"Score crédit   : {result['score']}/850")
        print(f"Niveau risque  : {result['risk_band']}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
