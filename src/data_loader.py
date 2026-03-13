"""
src/data_loader.py
------------------
Télécharge et prépare le Home Credit Default Risk Dataset.
Gère automatiquement 3 sources : Kaggle API, fichiers locaux, ou données synthétiques.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR    = Path("data/raw")
PROC_DIR   = Path("data/processed")

# Fichiers attendus depuis Kaggle
KAGGLE_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
]

# ─────────────────────────────────────────────────────────────────
# 1. ACQUISITION DES DONNÉES
# ─────────────────────────────────────────────────────────────────

def download_via_kaggle() -> bool:
    """Tente de télécharger via l'API Kaggle (~2.7 GB)."""
    try:
        import kaggle  # noqa
        print("📥 Téléchargement via Kaggle API…")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        os.system(
            f"kaggle competitions download -c home-credit-default-risk -p {RAW_DIR} --quiet"
        )
        zip_path = RAW_DIR / "home-credit-default-risk.zip"
        if zip_path.exists():
            print("📦 Extraction…")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(RAW_DIR)
            zip_path.unlink()
            print("✅ Données Kaggle extraites.")
            return True
    except Exception as e:
        print(f"⚠️  Kaggle indisponible : {e}")
    return False


def check_local_files() -> bool:
    """Vérifie si application_train.csv est déjà dans data/raw/."""
    main = RAW_DIR / "application_train.csv"
    if main.exists():
        print(f"✅ Fichier local détecté : {main}")
        return True
    return False


def generate_synthetic(n: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """
    Génère un dataset synthétique réaliste calqué sur Home Credit.
    Utilisé quand ni Kaggle ni fichiers locaux ne sont disponibles.
    """
    print(f"🔧 Génération de {n:,} observations synthétiques…")
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()
    df["SK_ID_CURR"] = np.arange(100_001, 100_001 + n)

    # --- Variables continues ---
    df["AMT_CREDIT"]       = rng.lognormal(11.8, 0.7, n).clip(45_000, 4_050_000).round(-3)
    df["AMT_INCOME_TOTAL"] = rng.lognormal(11.3, 0.6, n).clip(25_650, 1_575_000).round(-3)
    df["AMT_ANNUITY"]      = (df["AMT_CREDIT"] / rng.uniform(12, 60, n)).round(2)
    df["AMT_GOODS_PRICE"]  = (df["AMT_CREDIT"] * rng.uniform(0.8, 1.0, n)).round(-3)
    df["DAYS_BIRTH"]       = -(rng.integers(6_600, 25_000, n))   # négatif = jours avant aujourd'hui
    df["DAYS_EMPLOYED"]    = -(rng.integers(0, 10_000, n))
    df["DAYS_REGISTRATION"]= -(rng.integers(0, 20_000, n))
    df["DAYS_ID_PUBLISH"]  = -(rng.integers(0, 6_000, n))
    df["CNT_FAM_MEMBERS"]  = rng.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.45, 0.25, 0.10, 0.05])
    df["CNT_CHILDREN"]     = rng.choice([0, 1, 2, 3], n, p=[0.60, 0.25, 0.12, 0.03])
    df["EXT_SOURCE_1"]     = rng.uniform(0.0, 1.0, n)
    df["EXT_SOURCE_2"]     = rng.uniform(0.0, 1.0, n)
    df["EXT_SOURCE_3"]     = rng.uniform(0.0, 1.0, n)
    df["REGION_POPULATION_RELATIVE"] = rng.uniform(0.0, 0.07, n)
    df["HOUR_APPR_PROCESS_START"]    = rng.integers(0, 24, n)
    df["OBS_30_CNT_SOCIAL_CIRCLE"]   = rng.integers(0, 10, n)
    df["DEF_30_CNT_SOCIAL_CIRCLE"]   = rng.integers(0, 3, n)
    df["AMT_REQ_CREDIT_BUREAU_YEAR"] = rng.integers(0, 10, n).astype(float)

    # --- Variables catégorielles ---
    df["NAME_CONTRACT_TYPE"] = rng.choice(
        ["Cash loans", "Revolving loans"], n, p=[0.91, 0.09])
    df["CODE_GENDER"]  = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    df["FLAG_OWN_CAR"] = rng.choice(["Y", "N"], n, p=[0.34, 0.66])
    df["FLAG_OWN_REALTY"] = rng.choice(["Y", "N"], n, p=[0.69, 0.31])
    df["NAME_TYPE_SUITE"] = rng.choice(
        ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A"],
        n, p=[0.81, 0.11, 0.05, 0.02, 0.01])
    df["NAME_INCOME_TYPE"] = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        n, p=[0.52, 0.23, 0.18, 0.06, 0.01])
    df["NAME_EDUCATION_TYPE"] = rng.choice(
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
        n, p=[0.71, 0.18, 0.07, 0.03, 0.01])
    df["NAME_FAMILY_STATUS"] = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        n, p=[0.64, 0.16, 0.10, 0.07, 0.03])
    df["NAME_HOUSING_TYPE"] = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        n, p=[0.89, 0.04, 0.04, 0.02, 0.005, 0.005])
    df["OCCUPATION_TYPE"] = rng.choice(
        ["Laborers", "Core staff", "Accountants", "Managers",
         "Drivers", "Sales staff", "Cleaning staff", "Cooking staff",
         "Medicine staff", "Security staff", "High skill tech staff",
         "Waiters/barmen staff", "Low-skill Laborers", "Realty agents",
         "Secretaries", "IT staff", "HR staff", "Private service staff"],
        n)
    df["ORGANIZATION_TYPE"] = rng.choice(
        ["Business Entity Type 3", "School", "Government", "Religion",
         "Other", "Medicine", "Business Entity Type 2", "Self-employed",
         "Transport: type 2", "Construction", "Housing", "Kindergarten",
         "Trade: type 7", "Industry: type 11", "Military", "Services",
         "Security Ministries", "Transport: type 4", "Industry: type 1",
         "Emergency", "Security", "Trade: type 2", "University",
         "Transport: type 3", "Police", "Business Entity Type 1",
         "Postal", "Industry: type 4", "Agriculture", "Restaurant",
         "Culture", "Hotel", "Industry: type 7", "Trade: type 3",
         "Industry: type 3", "Bank", "Industry: type 9", "Insurance",
         "Trade: type 6", "Industry: type 2", "Transport: type 1",
         "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5",
         "Industry: type 10", "Legal Services", "Advertising",
         "Trade: type 4", "Electricity", "Trade: type 5", "Industry: type 6",
         "Industry: type 8", "XNA"], n)
    df["WEEKDAY_APPR_PROCESS_START"] = rng.choice(
        ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"], n)
    df["REGION_RATING_CLIENT"] = rng.choice([1, 2, 3], n, p=[0.09, 0.74, 0.17])
    df["REG_REGION_NOT_LIVE_REGION"] = rng.integers(0, 2, n)
    df["REG_CITY_NOT_LIVE_CITY"]     = rng.integers(0, 2, n)
    df["FLAG_EMAIL"] = rng.integers(0, 2, n)
    df["FLAG_PHONE"] = rng.integers(0, 2, n)
    df["FLAG_WORK_PHONE"] = rng.integers(0, 2, n)
    df["FLAG_DOCUMENT_3"] = rng.choice([0, 1], n, p=[0.50, 0.50])
    df["LIVE_CITY_NOT_WORK_CITY"] = rng.integers(0, 2, n)

    # --- Cible (logique réaliste ~8% de défaut) ---
    risk = (
        0.05
        - 0.06 * (df["EXT_SOURCE_2"] > 0.6).astype(float)
        - 0.04 * (df["EXT_SOURCE_3"] > 0.6).astype(float)
        + 0.05 * (df["DAYS_EMPLOYED"] > -365).astype(float)
        + 0.03 * (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"] > 5).astype(float)
        + 0.04 * (df["NAME_INCOME_TYPE"] == "Unemployed").astype(float)
        - 0.03 * (df["FLAG_OWN_REALTY"] == "Y").astype(float)
        + rng.uniform(-0.02, 0.08, n)
    ).clip(0.01, 0.35)
    df["TARGET"] = (rng.uniform(0, 1, n) < risk).astype(int)

    print(f"✅ Synthétique généré — défauts : {df['TARGET'].mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées les plus importantes."""
    df = df.copy()

    # Ratios financiers
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"]   = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"].replace(0, np.nan) + 1)

    # Âge & ancienneté
    df["AGE_YEARS"]      = (-df["DAYS_BIRTH"]) / 365.25
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"].clip(upper=0)) / 365.25
    df["EMPLOYED_RATIO"] = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1)

    # Score externe moyen
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCES_MIN"]  = df[ext_cols].min(axis=1)
    df["EXT_SOURCES_STD"]  = df[ext_cols].std(axis=1).fillna(0)

    # Charges sociales
    if "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["SOCIAL_CIRCLE_DEF_RATE"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] /
            (df["OBS_30_CNT_SOCIAL_CIRCLE"].replace(0, np.nan) + 1)
        ).fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode les variables catégorielles."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "SK_ID_CURR"]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=False)
    return df_enc


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs aberrantes et impute les NaN."""
    df = df.copy()
    # DAYS_EMPLOYED = 365243 = 'non-employé' dans Home Credit
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0)
    # Imputation médiane pour les numériques
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────────────────────────
# 3. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def load_or_generate() -> pd.DataFrame:
    """Charge les données depuis la meilleure source disponible."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Priorité 1 : fichier local déjà présent
    if check_local_files():
        print("📂 Chargement depuis data/raw/application_train.csv…")
        df = pd.read_csv(RAW_DIR / "application_train.csv")
        print(f"✅ {len(df):,} lignes chargées depuis Kaggle.")
        return df

    # Priorité 2 : Kaggle API
    if download_via_kaggle() and check_local_files():
        df = pd.read_csv(RAW_DIR / "application_train.csv")
        print(f"✅ {len(df):,} lignes chargées depuis Kaggle.")
        return df

    # Priorité 3 : données synthétiques
    return generate_synthetic()


def run_pipeline(sample_size: int = None) -> tuple:
    """
    Pipeline complet.
    Retourne (df_raw, df_encoded, feature_names).
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger
    df_raw = load_or_generate()

    # Optionnel : sous-échantillon pour développement rapide
    if sample_size and len(df_raw) > sample_size:
        df_raw = df_raw.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"🔀 Sous-échantillon : {sample_size:,} lignes")

    # 2. Sauvegarder brut décodé
    df_raw.to_csv(PROC_DIR / "home_credit_raw.csv", index=False)

    # 3. Nettoyage
    df_clean = clean_and_impute(df_raw)

    # 4. Feature engineering
    df_feat = engineer_features(df_clean)

    # 5. Encodage
    df_enc = encode_categoricals(df_feat)

    # 6. Supprimer SK_ID_CURR et TARGET des features
    feature_cols = [c for c in df_enc.columns if c not in ("SK_ID_CURR", "TARGET")]
    df_enc[feature_cols + ["TARGET"]].to_csv(PROC_DIR / "home_credit_encoded.csv", index=False)

    # Stats
    target_rate = df_raw["TARGET"].mean() * 100
    print(f"\n📊 Dataset final : {df_enc.shape[0]:,} lignes × {len(feature_cols)} features")
    print(f"   Taux de défaut : {target_rate:.2f}%")
    print(f"   Défauts : {df_raw['TARGET'].sum():,} / {len(df_raw):,}")

    return df_raw, df_enc, feature_cols


if __name__ == "__main__":
    run_pipeline()
