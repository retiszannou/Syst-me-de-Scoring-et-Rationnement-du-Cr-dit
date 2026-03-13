"""
tests/test_pipeline.py
----------------------
Tests unitaires — data_loader + predict
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

class TestSyntheticData:

    def test_shape(self):
        from data_loader import generate_synthetic
        df = generate_synthetic(n=200)
        assert df.shape[0] == 200
        assert "TARGET" in df.columns

    def test_target_binary(self):
        from data_loader import generate_synthetic
        df = generate_synthetic(n=500)
        assert set(df["TARGET"].unique()).issubset({0, 1})

    def test_default_rate_realistic(self):
        from data_loader import generate_synthetic
        df = generate_synthetic(n=2_000)
        rate = df["TARGET"].mean()
        assert 0.03 <= rate <= 0.30, f"Taux de défaut irréaliste : {rate:.2%}"

    def test_no_null_after_clean(self):
        from data_loader import generate_synthetic, clean_and_impute
        df = generate_synthetic(n=300)
        df_clean = clean_and_impute(df)
        num_nulls = df_clean.select_dtypes(include=[np.number]).isnull().sum().sum()
        assert num_nulls == 0

    def test_feature_engineering_adds_columns(self):
        from data_loader import generate_synthetic, clean_and_impute, engineer_features
        df = generate_synthetic(n=100)
        df = clean_and_impute(df)
        df = engineer_features(df)
        for col in ["CREDIT_INCOME_RATIO", "AGE_YEARS", "EXT_SOURCES_MEAN"]:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_encoding_no_objects(self):
        from data_loader import generate_synthetic, clean_and_impute, engineer_features, encode_categoricals
        df = generate_synthetic(n=200)
        df = clean_and_impute(df)
        df = engineer_features(df)
        df = encode_categoricals(df)
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        assert len(obj_cols) == 0, f"Colonnes object restantes : {obj_cols}"


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

class TestScoreConversion:

    def test_score_bounds(self):
        from predict import probability_to_score
        assert probability_to_score(0.0) == 850
        assert probability_to_score(1.0) == 300

    def test_score_monotone(self):
        from predict import probability_to_score
        probs = np.linspace(0, 1, 50)
        scores = [probability_to_score(p) for p in probs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Score non monotone à p={probs[i]:.2f}"

    def test_score_in_range(self):
        from predict import probability_to_score
        for p in np.linspace(0, 1, 20):
            s = probability_to_score(p)
            assert 300 <= s <= 850

    def test_risk_bands(self):
        from predict import score_to_risk_band
        assert score_to_risk_band(800)[0] == "FAIBLE"
        assert score_to_risk_band(700)[0] == "MODÉRÉ"
        assert score_to_risk_band(620)[0] == "ÉLEVÉ"
        assert score_to_risk_band(450)[0] == "TRÈS ÉLEVÉ"


class TestAlignFeatures:

    def test_adds_missing_columns(self):
        from predict import align_features
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = align_features(df, ["a", "b", "c", "d"])
        assert "c" in result.columns
        assert result["c"].iloc[0] == 0

    def test_removes_extra_columns(self):
        from predict import align_features
        df = pd.DataFrame({"a": [1], "b": [2], "EXTRA": [99]})
        result = align_features(df, ["a", "b"])
        assert "EXTRA" not in result.columns

    def test_column_order(self):
        from predict import align_features
        df = pd.DataFrame({"b": [2], "a": [1], "c": [3]})
        result = align_features(df, ["a", "b", "c"])
        assert list(result.columns) == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
