"""
app.py  —  CréditScope
Application Streamlit de scoring et de rationnement du crédit.
Dataset : Home Credit Default Risk (Kaggle)
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, "src")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CréditScope",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono&display=swap');
*, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'DM Mono', monospace !important; }

/* Cards décision */
.card-green {
    background: linear-gradient(135deg,#082b12,#0f3d1c);
    border:2px solid #27ae60; border-radius:18px;
    padding:28px 32px; text-align:center;
}
.card-red {
    background: linear-gradient(135deg,#2b0808,#3d0f0f);
    border:2px solid #c0392b; border-radius:18px;
    padding:28px 32px; text-align:center;
}
.big-score {
    font-size:3.6rem; font-weight:800;
    font-family:'DM Mono',monospace; letter-spacing:-2px;
}
.tag {
    display:inline-block; border-radius:99px;
    padding:4px 14px; font-size:.8rem; font-weight:600;
    margin:4px 2px;
}
.risk-low   { background:#0f3d1c; color:#2ecc71; border:1px solid #27ae60; }
.risk-med   { background:#3d2e07; color:#f39c12; border:1px solid #d68910; }
.risk-high  { background:#3d1c07; color:#e67e22; border:1px solid #ca6f1e; }
.risk-vhigh { background:#2b0808; color:#e74c3c; border:1px solid #c0392b; }
.section-title {
    font-size:1.1rem; font-weight:700; color:#8892a4;
    margin:18px 0 8px; padding-bottom:6px;
    border-bottom:1px solid #1e2a3a;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def get_data():
    p = Path("data/processed/home_credit_raw.csv")
    if not p.exists():
        from data_loader import run_pipeline
        run_pipeline()
    return pd.read_csv(p)


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model_bundle():
    models_dir = Path("models")
    if not (models_dir / "best_model.pkl").exists():
        from train import run_training
        run_training()
    model        = joblib.load(models_dir / "best_model.pkl")
    metrics      = json.loads((models_dir / "metrics.json").read_text())
    feat_names   = json.loads((models_dir / "feature_names.json").read_text())
    best_name    = (models_dir / "best_model_name.txt").read_text().strip()
    return model, metrics, feat_names, best_name


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 CréditScope")
    st.caption("Home Credit Default Risk · ML Scoring")
    st.divider()

    page = st.radio("", [
        "🎯 Évaluation d'un Profil",
        "📊 Exploration des Données",
        "🤖 Performance des Modèles",
        "📖 À propos",
    ], label_visibility="collapsed")

    st.divider()
    try:
        _, metrics, _, best_name = get_model_bundle()
        bm = metrics[best_name]
        st.success(f"✅ {best_name.replace('_',' ').title()}")
        c1, c2 = st.columns(2)
        c1.metric("AUC-ROC",  f"{bm['roc_auc']:.3f}")
        c2.metric("Accuracy", f"{bm['accuracy']:.1%}")
    except Exception:
        st.warning("Modèle non chargé.\nLancez `python src/train.py`")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — ÉVALUATION D'UN PROFIL
# ═══════════════════════════════════════════════════════════════
if page == "🎯 Évaluation d'un Profil":
    st.title("🎯 Évaluation d'un Demandeur de Crédit")
    st.markdown("Renseignez le profil et obtenez une décision d'octroi ou de rationnement.")

    with st.form("form_credit"):

        # ── Ligne 1 ──
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-title">💰 Informations Financières</div>',
                        unsafe_allow_html=True)
            amt_credit       = st.number_input("Montant du crédit (USD)", 10_000, 4_050_000, 202_500, 5_000)
            amt_income       = st.number_input("Revenu annuel (USD)", 10_000, 1_575_000, 67_500, 1_000)
            amt_annuity      = st.number_input("Mensualité (USD)", 500, 250_000, 7_956, 100)
            amt_goods        = st.number_input("Prix du bien financé (USD)", 0, 4_050_000, 180_000, 5_000)
            contract_type    = st.selectbox("Type de contrat",
                                            ["Cash loans", "Revolving loans"])

        with col2:
            st.markdown('<div class="section-title">👤 Profil Personnel</div>',
                        unsafe_allow_html=True)
            age_years        = st.slider("Âge (ans)", 18, 70, 33)
            gender           = st.selectbox("Genre", ["F", "M"])
            family_status    = st.selectbox("Statut familial",
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
            cnt_children     = st.number_input("Enfants à charge", 0, 10, 0)
            cnt_fam          = st.number_input("Membres du foyer", 1, 10, 2)
            own_car          = st.selectbox("Propriétaire d'un véhicule", ["N", "Y"])
            own_realty       = st.selectbox("Propriétaire immobilier", ["Y", "N"])
            housing_type     = st.selectbox("Type de logement",
                ["House / apartment", "With parents", "Municipal apartment",
                 "Rented apartment", "Office apartment", "Co-op apartment"])

        with col3:
            st.markdown('<div class="section-title">🏢 Situation Professionnelle</div>',
                        unsafe_allow_html=True)
            income_type      = st.selectbox("Source de revenus",
                ["Working", "Commercial associate", "Pensioner",
                 "State servant", "Unemployed"])
            education        = st.selectbox("Niveau d'éducation",
                ["Secondary / secondary special", "Higher education",
                 "Incomplete higher", "Lower secondary", "Academic degree"])
            employed_years   = st.slider("Ancienneté emploi (ans)", 0, 40, 5)
            occupation       = st.selectbox("Profession",
                ["Laborers", "Core staff", "Accountants", "Managers",
                 "Drivers", "Sales staff", "Cleaning staff",
                 "Medicine staff", "Security staff", "High skill tech staff",
                 "IT staff", "HR staff", "Secretaries", "Realty agents"])
            organization     = st.selectbox("Secteur d'activité",
                ["Business Entity Type 3", "Self-employed", "Government",
                 "School", "Medicine", "Bank", "Construction",
                 "Transport: type 2", "Police", "Military", "Other"])

        st.divider()

        # ── Ligne 2 ──
        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="section-title">📈 Scores & Historique</div>',
                        unsafe_allow_html=True)
            ext1 = st.slider("Score externe 1", 0.0, 1.0, 0.60, 0.01)
            ext2 = st.slider("Score externe 2", 0.0, 1.0, 0.72, 0.01)
            ext3 = st.slider("Score externe 3", 0.0, 1.0, 0.55, 0.01)
            region_rating = st.selectbox("Rating région", [1, 2, 3], index=1)

        with col5:
            st.markdown('<div class="section-title">📋 Divers</div>',
                        unsafe_allow_html=True)
            flag_phone      = st.selectbox("Téléphone fixe", [1, 0], format_func=lambda x: "Oui" if x else "Non")
            flag_email      = st.selectbox("Email enregistré", [1, 0], format_func=lambda x: "Oui" if x else "Non")
            flag_work_phone = st.selectbox("Téléphone professionnel", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            def_social      = st.slider("Défauts entourage (30j)", 0, 10, 0)
            obs_social      = st.slider("Observations entourage (30j)", 0, 20, 2)
            req_bureau_year = st.slider("Demandes bureau crédit / an", 0, 20, 1)

        submitted = st.form_submit_button("🔍 Analyser le Profil",
                                          use_container_width=True, type="primary")

    # ── Résultats ──────────────────────────────────────────────
    if submitted:
        from predict import predict, EXAMPLE_PROFILE

        profile = {
            **EXAMPLE_PROFILE,   # valeurs par défaut pour colonnes non saisies
            "AMT_CREDIT":        amt_credit,
            "AMT_INCOME_TOTAL":  amt_income,
            "AMT_ANNUITY":       amt_annuity,
            "AMT_GOODS_PRICE":   amt_goods,
            "DAYS_BIRTH":        -(age_years * 365),
            "DAYS_EMPLOYED":     -(employed_years * 365),
            "CNT_CHILDREN":      cnt_children,
            "CNT_FAM_MEMBERS":   cnt_fam,
            "EXT_SOURCE_1":      ext1,
            "EXT_SOURCE_2":      ext2,
            "EXT_SOURCE_3":      ext3,
            "NAME_CONTRACT_TYPE": contract_type,
            "CODE_GENDER":       gender,
            "FLAG_OWN_CAR":      own_car,
            "FLAG_OWN_REALTY":   own_realty,
            "NAME_INCOME_TYPE":  income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": housing_type,
            "OCCUPATION_TYPE":   occupation,
            "ORGANIZATION_TYPE": organization,
            "REGION_RATING_CLIENT": region_rating,
            "FLAG_PHONE":        flag_phone,
            "FLAG_EMAIL":        flag_email,
            "FLAG_WORK_PHONE":   flag_work_phone,
            "DEF_30_CNT_SOCIAL_CIRCLE": def_social,
            "OBS_30_CNT_SOCIAL_CIRCLE": obs_social,
            "AMT_REQ_CREDIT_BUREAU_YEAR": float(req_bureau_year),
        }

        try:
            model, _, feat_names, _ = get_model_bundle()

            with st.spinner("Calcul du score…"):
                result = predict(profile, "best")

            st.divider()
            col_d, col_g, col_f = st.columns([2, 1.2, 2])

            # Carte décision
            with col_d:
                if result["decision"] == "ACCORDER":
                    css = "card-green"
                    icon, label, col_txt = "✅", "CRÉDIT ACCORDÉ", "#2ecc71"
                else:
                    css = "card-red"
                    icon, label, col_txt = "❌", "CRÉDIT RATIONNÉ", "#e74c3c"

                risk_css = {
                    "FAIBLE": "risk-low", "MODÉRÉ": "risk-med",
                    "ÉLEVÉ": "risk-high", "TRÈS ÉLEVÉ": "risk-vhigh"
                }[result["risk_band"]]

                st.markdown(f"""
                <div class="{css}">
                  <div style="font-size:2.8rem">{icon}</div>
                  <div class="big-score" style="color:{col_txt}">{label}</div>
                  <br>
                  <span class="tag {risk_css}">Risque {result['risk_band']}</span>
                  <div style="margin-top:12px;color:{col_txt};font-size:1rem">
                    P(défaut) = <strong>{result['prob_default']*100:.1f}%</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Métriques
            with col_g:
                st.metric("Score Crédit",  f"{result['score']} / 850")
                st.metric("P(remboursement)", f"{result['prob_repay']*100:.1f}%")
                st.metric("P(défaut)",        f"{result['prob_default']*100:.1f}%")
                ratio = amt_credit / max(amt_income, 1)
                st.metric("Ratio crédit/revenu", f"{ratio:.2f}×")

            # Gauge
            with col_f:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["score"],
                    number={"suffix": " / 850", "font": {"size": 22}},
                    title={"text": "Score Crédit", "font": {"size": 13}},
                    gauge={
                        "axis":  {"range": [300, 850]},
                        "bar":   {"color": result["risk_color"], "thickness": 0.28},
                        "bgcolor": "#111827",
                        "steps": [
                            {"range": [300, 580], "color": "#2d0e0e"},
                            {"range": [580, 670], "color": "#2d1e0e"},
                            {"range": [670, 740], "color": "#1a2a0e"},
                            {"range": [740, 850], "color": "#0e2a18"},
                        ],
                        "threshold": {
                            "line": {"color": result["risk_color"], "width": 4},
                            "thickness": 0.75,
                            "value": result["score"],
                        },
                    },
                ))
                fig.update_layout(
                    height=230, margin=dict(t=40, b=10, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recommandation
            st.divider()
            if result["decision"] == "RATIONNER":
                st.error("""
                **Recommandations pour améliorer le dossier :**
                - 🔻 Réduire le montant demandé (ratio crédit/revenu trop élevé)
                - ⏳ Augmenter la durée pour réduire la mensualité
                - 🏦 Apporter des garanties supplémentaires (co-emprunteur, caution)
                - 📈 Améliorer les scores externes (historique de remboursement)
                """)
            else:
                st.success(
                    f"✅ **Profil approuvé.** Score **{result['score']}/850** "
                    f"— Risque {result['risk_band']} "
                    f"— P(défaut) : {result['prob_default']*100:.1f}%"
                )

        except FileNotFoundError:
            st.error("Modèle non disponible. Lancez `python src/train.py` puis relancez l'app.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Exploration des Données":
    st.title("📊 Exploration — Home Credit Default Risk")

    try:
        df = get_data()

        # KPIs
        n_total   = len(df)
        n_default = df["TARGET"].sum()
        n_good    = n_total - n_default

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total demandes",   f"{n_total:,}")
        c2.metric("Sans défaut",      f"{n_good:,}",    f"{n_good/n_total*100:.1f}%")
        c3.metric("En défaut",        f"{n_default:,}", f"-{n_default/n_total*100:.1f}%",
                  delta_color="inverse")
        c4.metric("Taux de défaut",   f"{n_default/n_total*100:.2f}%")
        c5.metric("Variables",        str(df.shape[1] - 1))

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Distribution", "Profil financier", "Démographie", "Données brutes"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.pie(
                    values=[n_good, n_default],
                    names=["Pas de défaut", "Défaut"],
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                    title="Répartition TARGET (défaut de crédit)",
                    hole=0.45,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig = px.histogram(
                    df, x="EXT_SOURCE_2",
                    color=df["TARGET"].map({0: "Pas de défaut", 1: "Défaut"}),
                    nbins=40, barmode="overlay",
                    color_discrete_map={"Pas de défaut": "#2ecc71", "Défaut": "#e74c3c"},
                    title="Score Externe 2 par TARGET",
                    labels={"color": "TARGET"},
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.box(
                    df, x=df["TARGET"].map({0: "Pas de défaut", 1: "Défaut"}),
                    y="AMT_CREDIT",
                    color=df["TARGET"].map({0: "Pas de défaut", 1: "Défaut"}),
                    color_discrete_map={"Pas de défaut": "#2ecc71", "Défaut": "#e74c3c"},
                    title="Montant du crédit vs TARGET",
                    labels={"x": "TARGET", "AMT_CREDIT": "Montant (USD)"},
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
                fig = px.histogram(
                    df[df["CREDIT_INCOME_RATIO"] < 20],
                    x="CREDIT_INCOME_RATIO",
                    color=df["TARGET"].map({0: "Pas de défaut", 1: "Défaut"}),
                    nbins=50, barmode="overlay",
                    color_discrete_map={"Pas de défaut": "#3498db", "Défaut": "#e74c3c"},
                    title="Ratio Crédit / Revenu par TARGET",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                # Taux de défaut par type de revenu
                grp = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=grp.values * 100, y=grp.index,
                    orientation="h",
                    title="Taux de défaut par source de revenus (%)",
                    color=grp.values,
                    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                    labels={"x": "Taux de défaut (%)", "y": "Source de revenus"},
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Distribution âge
                df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
                fig = px.histogram(
                    df, x="AGE_YEARS",
                    color=df["TARGET"].map({0: "Pas de défaut", 1: "Défaut"}),
                    nbins=30, barmode="overlay",
                    color_discrete_map={"Pas de défaut": "#3498db", "Défaut": "#e74c3c"},
                    title="Distribution de l'Âge par TARGET",
                    labels={"AGE_YEARS": "Âge (ans)", "color": "TARGET"},
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"100 premières lignes sur {len(df):,}")

    except Exception as e:
        st.error(f"Erreur : {e}")
        st.info("Lancez `python src/data_loader.py` pour préparer les données.")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE DES MODÈLES
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Performance des Modèles":
    st.title("🤖 Performance des Modèles ML")

    try:
        _, metrics, feat_names, best_name = get_model_bundle()

        # Tableau de comparaison
        rows = []
        for n, m in metrics.items():
            rows.append({
                "Modèle":       n.replace("_", " ").title() + (" 🏆" if n == best_name else ""),
                "ROC-AUC":      round(m["roc_auc"], 4),
                "Avg Precision":round(m["avg_precision"], 4),
                "Accuracy":     round(m["accuracy"], 4),
                "F1-Score":     round(m["f1_score"], 4),
            })
        st.subheader("Comparaison des modèles")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            # Courbes ROC
            colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
            fig_roc = go.Figure()
            for i, (n, m) in enumerate(metrics.items()):
                if "roc_curve" in m:
                    lbl = n.replace("_", " ").title()
                    if n == best_name:
                        lbl += " 🏆"
                    fig_roc.add_trace(go.Scatter(
                        x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                        name=f"{lbl} ({m['roc_auc']:.3f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Aléatoire", line=dict(dash="dash", color="gray"),
            ))
            fig_roc.update_layout(
                title="Courbes ROC", height=400,
                xaxis_title="FPR", yaxis_title="TPR",
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                legend=dict(bgcolor="rgba(20,25,40,.8)"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_right:
            # Matrice de confusion
            bm  = metrics[best_name]
            cm  = np.array(bm["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Prédit Non-défaut", "Prédit Défaut"],
                y=["Réel Non-défaut",   "Réel Défaut"],
                color_continuous_scale=[[0, "#111827"], [1, "#2ecc71"]],
                title=f"Matrice de confusion — {best_name.replace('_',' ').title()}",
            )
            fig_cm.update_layout(height=400,
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_cm, use_container_width=True)

        # Importance des variables
        if bm.get("feature_importance"):
            st.subheader("Top 20 variables — importance")
            top = dict(list(bm["feature_importance"].items())[:20])
            fig_fi = px.bar(
                x=list(top.values()), y=list(top.keys()),
                orientation="h",
                color=list(top.values()),
                color_continuous_scale="Tealgrn",
                labels={"x": "Importance", "y": "Variable"},
                title="",
            )
            fig_fi.update_layout(
                height=520, yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")
        st.info("Lancez `python src/train.py`")


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — À PROPOS
# ═══════════════════════════════════════════════════════════════
elif page == "📖 À propos":
    st.title("📖 À propos du projet")
    st.markdown("""
## 🏦 CréditScope — Scoring & Rationnement du Crédit

Système de scoring crédit par Machine Learning basé sur le **Home Credit Default Risk Dataset** (Kaggle, ~307 000 demandes, 122 variables).

---

### 🎯 Problématique économique

Le **rationnement du crédit** (Stiglitz & Weiss, 1981) désigne le refus d'accorder un crédit à un emprunteur, même solvable, en raison de l'asymétrie d'information.
Ce modèle aide à objectiver la décision en calculant un **score de risque** basé sur des données comportementales et financières.

---

### 📁 Fichiers Home Credit utilisés

| Fichier | Description |
|---|---|
| `application_train.csv` | Profils des demandeurs + TARGET |
| `bureau.csv` | Historique bureau de crédit |
| `previous_application.csv` | Demandes passées |
| `installments_payments.csv` | Historique de remboursement |

---

### 🤖 Modèles

| Modèle | Spécificité |
|---|---|
| Logistic Regression | Baseline interprétable |
| Random Forest | Robuste aux outliers |
| LightGBM | Meilleure performance (AUC ~0.76) |
| XGBoost | Concurrent sérieux |

---

### ⚙️ Installation rapide

```bash
git clone https://github.com/votre-username/credit-scoring-home-credit
cd credit-scoring-home-credit
pip install -r requirements.txt

# Option A : avec données Kaggle
kaggle competitions download -c home-credit-default-risk -p data/raw
python src/train.py

# Option B : sans Kaggle (données synthétiques)
python src/train.py --sample 10000

streamlit run app.py
```

---

### 📚 Références

- Stiglitz, J.E. & Weiss, A. (1981). *Credit Rationing in Markets with Imperfect Information*. AER.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.
- Home Credit Group (2018). [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

---

> ⚠️ **Projet à but éducatif.** Les décisions réelles doivent respecter le RGPD et les réglementations bancaires (équité algorithmique, non-discrimination).
""")
 
 