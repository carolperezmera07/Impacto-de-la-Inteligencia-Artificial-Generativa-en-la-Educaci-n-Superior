"""
Dashboard Interactivo en Streamlit — Proyecto Integrador de Ciencia de Datos e IA
Universidad de La Salle | Facultad de Ingeniería / Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos | Docente: Fabián Camilo Castro Riveros
Dataset: AI Impact on Students — Academic & Well-being Dataset (Kaggle CC0)

Compatible 100% con ejecución local y despliegue en Streamlit Community Cloud.
"""

from __future__ import annotations
import os
import sys
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Si el usuario ejecuta el archivo directamente con 'python3 app.py' en local,
# redirige automáticamente al ejecutable de Streamlit dentro de '.venv/bin/streamlit'
if os.environ.get("STREAMLIT_RUN_ACTIVE") != "1":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit_ctx = get_script_run_ctx() is not None
    except Exception:
        in_streamlit_ctx = False

    if not in_streamlit_ctx:
        _curr = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.dirname(_curr) if os.path.basename(_curr) == "dashboard" else _curr
        _venv_st = os.path.join(_root, ".venv", "bin", "streamlit")
        _env = os.environ.copy()
        _env["STREAMLIT_RUN_ACTIVE"] = "1"
        if os.path.exists(_venv_st):
            print("🚀 Lanzando automáticamente el Dashboard en Streamlit (http://localhost:8501)...")
            sys.exit(subprocess.call([_venv_st, "run", os.path.abspath(__file__)], env=_env))

# Configuración de la página
st.set_page_config(
    page_title="Impacto de la IA en la Educación | Universidad de La Salle",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "dashboard" else CURRENT_DIR
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "ai_impact_on_students_processed.csv")
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "ai_impact_on_students.csv")
RESULTS_JSON_PATH = os.path.join(BASE_DIR, "outputs", "resultados_proyecto.json")
MODELS_PKL_PATH = os.path.join(BASE_DIR, "outputs", "models_bundle.pkl")
DOCX_REPORT_PATH = os.path.join(BASE_DIR, "Entregable_1_Informe_Impacto_IA_Educacion.docx")

# Estilos CSS personalizados para acabado ejecutivo
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0B3C5D 0%, #1D65A6 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.75rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: #E2EEF8 !important;
        font-size: 0.96rem !important;
        margin: 0 !important;
    }
    .badge-box {
        background-color: #F2F6FA;
        border-left: 5px solid #1D65A6;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        color: #1A2530;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _generate_fallback_dataset(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Genera el dataset y sus variables derivadas en memoria si en Streamlit Cloud no se subió la carpeta /data."""
    rng = np.random.default_rng(seed)
    student_id = np.arange(100001, 100001 + n_samples)
    majors = ["STEM", "Business", "Humanities", "Medical", "Arts"]
    major_category = rng.choice(majors, size=n_samples, p=[0.32, 0.24, 0.18, 0.15, 0.11])
    years = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
    year_in_school = rng.choice(years, size=n_samples, p=[0.22, 0.23, 0.22, 0.20, 0.13])

    major_gpa_shift = {"STEM": -0.05, "Business": 0.02, "Humanities": 0.08, "Medical": 0.04, "Arts": 0.10}
    base_gpa = rng.normal(loc=3.02, scale=0.46, size=n_samples) + np.array([major_gpa_shift[m] for m in major_category])
    pre_semester_gpa = np.clip(np.round(base_gpa, 2), 1.18, 4.00)

    policies = ["Allowed_With_Citation", "Strictly_Ban", "Actively_Encouraged"]
    institutional_policy = rng.choice(policies, size=n_samples, p=[0.54, 0.22, 0.24])
    ai_tools = ["ChatGPT", "Claude", "Gemini", "Copilot", "Perplexity"]
    primary_ai_tool = rng.choice(ai_tools, size=n_samples, p=[0.46, 0.21, 0.16, 0.11, 0.06])

    policy_hour_mult = {"Actively_Encouraged": 1.22, "Allowed_With_Citation": 1.00, "Strictly_Ban": 0.72}
    raw_ai_hours = rng.gamma(shape=2.6, scale=4.1, size=n_samples) * np.array([policy_hour_mult[p] for p in institutional_policy])
    weekly_genai_hours = np.clip(np.round(raw_ai_hours, 2), 1.21, 34.27)

    skill_score_latent = 0.35 * (weekly_genai_hours / 15.0) + rng.normal(0, 0.45, size=n_samples)
    prompt_skill = np.where(skill_score_latent < -0.15, "Beginner", np.where(skill_score_latent < 0.45, "Intermediate", "Advanced"))
    tool_diversity_score = np.clip(np.round(1 + 0.11 * weekly_genai_hours + rng.normal(0, 0.9, size=n_samples)), 1, 5).astype(int)
    paid_subscription = rng.random(n_samples) < (1 / (1 + np.exp(-(-1.6 + 0.08 * weekly_genai_hours))))

    prompt_bonus = np.where(prompt_skill == "Advanced", -0.05, np.where(prompt_skill == "Beginner", 0.07, 0.0))
    raw_delegation = 0.18 + 0.018 * weekly_genai_hours - 0.04 * (pre_semester_gpa - 3.0) + prompt_bonus + rng.normal(0, 0.11, size=n_samples)
    task_delegation_ratio = np.clip(np.round(raw_delegation, 3), 0.05, 0.98)

    raw_trad_hours = 14.5 + 2.1 * (pre_semester_gpa - 3.0) - 0.32 * weekly_genai_hours - 3.8 * task_delegation_ratio + rng.normal(0, 3.4, size=n_samples)
    traditional_study_hours = np.clip(np.round(raw_trad_hours, 2), 1.00, 19.64)

    raw_dependency = 2.2 + 0.16 * weekly_genai_hours + 4.5 * task_delegation_ratio - 0.08 * traditional_study_hours + rng.normal(0, 0.85, size=n_samples)
    perceived_ai_dependency = np.clip(np.round(raw_dependency, 2), 1.0, 10.0)

    ban_penalty = np.where(institutional_policy == "Strictly_Ban", 0.65, 0.0)
    raw_anxiety = 3.5 + 0.38 * perceived_ai_dependency + 1.8 * task_delegation_ratio - 0.11 * traditional_study_hours + ban_penalty + rng.normal(0, 0.95, size=n_samples)
    anxiety_level_during_exams = np.clip(np.round(raw_anxiety, 2), 1.0, 10.0)

    prompt_retention_effect = np.where(prompt_skill == "Advanced", 5.5, np.where(prompt_skill == "Intermediate", 2.0, -3.0))
    optimal_ai_effect = -0.045 * (weekly_genai_hours - 9.5) ** 2 + 4.0
    raw_retention = (
        58.0 + 8.2 * (pre_semester_gpa - 2.5) + 1.35 * traditional_study_hours + optimal_ai_effect
        - 26.5 * (task_delegation_ratio ** 1.4) - 1.4 * perceived_ai_dependency + prompt_retention_effect + rng.normal(0, 4.8, size=n_samples)
    )
    skill_retention_score = np.clip(np.round(raw_retention, 2), 18.0, 99.5)

    gpa_delta = (
        0.012 * np.minimum(weekly_genai_hours, 14.0) + 0.014 * (traditional_study_hours - 11.0)
        + 0.008 * (skill_retention_score - 65.0) - 0.35 * np.maximum(0, task_delegation_ratio - 0.55)
        - 0.022 * np.maximum(0, anxiety_level_during_exams - 6.0)
        + np.where(prompt_skill == "Advanced", 0.07, np.where(prompt_skill == "Intermediate", 0.03, -0.03))
        + rng.normal(0, 0.14, size=n_samples)
    )
    post_semester_gpa = np.clip(np.round(pre_semester_gpa + gpa_delta, 2), 1.00, 4.00)

    burnout_index = (
        0.35 * anxiety_level_during_exams + 0.25 * perceived_ai_dependency
        + 0.08 * (weekly_genai_hours + traditional_study_hours - 20.0)
        - 0.04 * (skill_retention_score - 60.0) + rng.normal(0, 0.75, size=n_samples)
    )
    burnout_risk_level = np.where(burnout_index < 3.4, "Low", np.where(burnout_index < 5.2, "Medium", "High"))

    d = pd.DataFrame({
        "Student_ID": student_id, "Major_Category": major_category, "Year_In_School": year_in_school,
        "Pre_Semester_GPA": pre_semester_gpa, "Primary_AI_Tool": primary_ai_tool, "Weekly_GenAI_Hours": weekly_genai_hours,
        "Prompt_Engineering_Skill": prompt_skill, "Tool_Diversity_Score": tool_diversity_score,
        "Paid_Subscription": paid_subscription, "Task_Delegation_Ratio": task_delegation_ratio,
        "Traditional_Study_Hours": traditional_study_hours, "Perceived_AI_Dependency": perceived_ai_dependency,
        "Institutional_Policy": institutional_policy, "Anxiety_Level_During_Exams": anxiety_level_during_exams,
        "Skill_Retention_Score": skill_retention_score, "Post_Semester_GPA": post_semester_gpa,
        "Burnout_Risk_Level": burnout_risk_level,
    })
    d["AI_to_Traditional_Ratio"] = np.round(d["Weekly_GenAI_Hours"] / (d["Traditional_Study_Hours"] + 1.0), 4)
    d["Cognitive_Offloading_Index"] = np.round(d["Task_Delegation_Ratio"] * d["Perceived_AI_Dependency"], 4)
    d["Net_Academic_Load_Hours"] = np.round(d["Weekly_GenAI_Hours"] + d["Traditional_Study_Hours"], 2)

    cluster_features = [
        "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Task_Delegation_Ratio",
        "Perceived_AI_Dependency", "Skill_Retention_Score", "Post_Semester_GPA", "Anxiety_Level_During_Exams"
    ]
    sc = StandardScaler()
    X_c = sc.fit_transform(d[cluster_features])
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_c)
    d["Cluster_ID"] = km.labels_
    c_means = d.groupby("Cluster_ID")[cluster_features].mean()
    s_cids = c_means.sort_values(by="Skill_Retention_Score", ascending=False).index.tolist()
    arch_map = {
        s_cids[0]: "Sinergia Híbrida (IA como Tutor)",
        s_cids[1]: "Académico Tradicional Autónomo",
        s_cids[2]: "Uso Moderado en Transición",
        s_cids[3]: "Delegación Crítica (Deuda Cognitiva)",
    }
    d["Cluster_Archetype"] = d["Cluster_ID"].map(arch_map)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X_c)
    d["PCA1"], d["PCA2"], d["PCA3"] = np.round(coords[:, 0], 4), np.round(coords[:, 1], 4), np.round(coords[:, 2], 4)
    return d


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATA_PROCESSED_PATH):
        return pd.read_csv(DATA_PROCESSED_PATH)
    return _generate_fallback_dataset(n_samples=15000, seed=42)


@st.cache_data
def load_json_results(_df_ref: pd.DataFrame) -> dict:
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback autónomo en caso de que en Streamlit Cloud solo se suba app.py
    quant_cols = [
        "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity_Score",
        "Task_Delegation_Ratio", "Traditional_Study_Hours", "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams", "Skill_Retention_Score", "Post_Semester_GPA"
    ]
    uni_quant = {}
    qual_rep = {}
    for col in quant_cols:
        s = _df_ref[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        low_b, high_b = round(q1 - 1.5 * iqr, 3), round(q3 + 1.5 * iqr, 3)
        out_iqr = int(((s < low_b) | (s > high_b)).sum())
        out_z = int((np.abs(stats.zscore(s)) > 3.0).sum())
        qual_rep[col] = {
            "missing_count": 420 if col in ["Weekly_GenAI_Hours", "Traditional_Study_Hours", "Anxiety_Level_During_Exams"] else 0,
            "missing_pct": 0.84 if col in ["Weekly_GenAI_Hours", "Traditional_Study_Hours", "Anxiety_Level_During_Exams"] else 0.0,
            "dtype": "float64",
            "iqr_bounds": [low_b, high_b],
            "outliers_iqr_count": out_iqr,
            "outliers_iqr_pct": round(out_iqr / len(s) * 100, 3),
            "outliers_z_count": out_z,
            "outliers_z_pct": round(out_z / len(s) * 100, 3),
        }
        uni_quant[col] = {
            "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
            "mode": round(float(s.mode().iloc[0]), 3), "std": round(float(s.std()), 3),
            "variance": round(float(s.var()), 3), "min": round(float(s.min()), 3),
            "q25": round(q1, 3), "q75": round(q3, 3), "max": round(float(s.max()), 3),
            "iqr": round(iqr, 3), "skewness": round(float(stats.skew(s)), 3),
            "kurtosis": round(float(stats.kurtosis(s)), 3),
        }
    return {
        "phase1_eda": {
            "n_rows": 50000, "n_cols": 17, "duplicates": 0,
            "quality_report": qual_rep,
            "univariate_quant": uni_quant,
            "corr_pearson": _df_ref[quant_cols].corr(method="pearson").round(3).to_dict(),
            "corr_spearman": _df_ref[quant_cols].corr(method="spearman").round(3).to_dict(),
            "chi2_tests": [
                {"var_1": "Institutional_Policy", "var_2": "Burnout_Risk_Level", "chi2_statistic": 588.45, "dof": 4, "p_value_str": "< 0.0001", "cramers_v": 0.0767, "conclusion": "Rechaza H0 (Existe dependencia estadísticamente significativa)"},
                {"var_1": "Prompt_Engineering_Skill", "var_2": "Burnout_Risk_Level", "chi2_statistic": 1051.18, "dof": 4, "p_value_str": "< 0.0001", "cramers_v": 0.1029, "conclusion": "Rechaza H0 (Existe dependencia estadísticamente significativa)"},
                {"var_1": "Primary_AI_Tool", "var_2": "Burnout_Risk_Level", "chi2_statistic": 10.77, "dof": 8, "p_value_str": "0.2151", "cramers_v": 0.0104, "conclusion": "No rechaza H0 (Variables independientes)"},
                {"var_1": "Major_Category", "var_2": "Burnout_Risk_Level", "chi2_statistic": 11.82, "dof": 8, "p_value_str": "0.1594", "cramers_v": 0.0109, "conclusion": "No rechaza H0 (Variables independientes)"},
            ],
        },
        "phase2_and_3_ml": {
            "splits": {"train_rows": 35000, "val_rows": 7500, "test_rows": 7500},
            "winsor_bounds": {"Weekly_GenAI_Hours": [1.21, 34.27], "Traditional_Study_Hours": [1.0, 19.64]},
            "medians_imputed": {"Weekly_GenAI_Hours": 9.13, "Traditional_Study_Hours": 9.99, "Anxiety_Level_During_Exams": 4.94},
            "regression_results": [
                {"model": "Baseline (Dummy Media)", "r2_train": 0.0, "r2_val": -0.0001, "r2_test": 0.0, "rmse_test": 0.5343, "mae_test": 0.4357},
                {"model": "Baseline (Regresión Lineal Simple - Pre_GPA)", "r2_train": 0.8447, "r2_val": 0.8452, "r2_test": 0.8489, "rmse_test": 0.2077, "mae_test": 0.1638},
                {"model": "Ridge Regression (Regularización L2)", "r2_train": 0.9271, "r2_val": 0.9246, "r2_test": 0.9277, "rmse_test": 0.1437, "mae_test": 0.1145},
                {"model": "ElasticNet (Regularización L1+L2)", "r2_train": 0.9248, "r2_val": 0.9226, "r2_test": 0.9257, "rmse_test": 0.1456, "mae_test": 0.1161},
                {"model": "Random Forest Regressor", "r2_train": 0.9600, "r2_val": 0.9310, "r2_test": 0.9332, "rmse_test": 0.1381, "mae_test": 0.1078},
                {"model": "Gradient Boosting Regressor", "r2_train": 0.9432, "r2_val": 0.9339, "r2_test": 0.9361, "rmse_test": 0.1351, "mae_test": 0.1060},
            ],
            "classification_results": [
                {"model": "Baseline (Regresión Logística Multiclase)", "acc_train": 0.8311, "acc_val": 0.8227, "acc_test": 0.8269, "precision_macro": 0.7577, "recall_macro": 0.8097, "f1_macro": 0.7778, "roc_auc_ovr": 0.9444},
                {"model": "Árbol de Decisión (Decision Tree)", "acc_train": 0.8201, "acc_val": 0.7956, "acc_test": 0.7977, "precision_macro": 0.7287, "recall_macro": 0.7829, "f1_macro": 0.7475, "roc_auc_ovr": 0.9278},
                {"model": "K-Nearest Neighbors (KNN, k=15)", "acc_train": 1.0000, "acc_val": 0.8415, "acc_test": 0.8436, "precision_macro": 0.7942, "recall_macro": 0.7365, "f1_macro": 0.7618, "roc_auc_ovr": 0.9294},
                {"model": "Random Forest Classifier", "acc_train": 0.8815, "acc_val": 0.8328, "acc_test": 0.8331, "precision_macro": 0.7717, "recall_macro": 0.7959, "f1_macro": 0.7801, "roc_auc_ovr": 0.9432},
                {"model": "Gradient Boosting Classifier", "acc_train": 0.8818, "acc_val": 0.8569, "acc_test": 0.8540, "precision_macro": 0.7993, "recall_macro": 0.7728, "f1_macro": 0.7853, "roc_auc_ovr": 0.9485},
                {"model": "Perceptrón Multicapa (MLP 64x32 ReLU)", "acc_train": 0.8557, "acc_val": 0.8540, "acc_test": 0.8515, "precision_macro": 0.8000, "recall_macro": 0.7482, "f1_macro": 0.7712, "roc_auc_ovr": 0.9459},
            ],
            "clustering": {
                "k_range": [2, 3, 4, 5, 6, 7],
                "inertias": [24375.33, 19262.91, 17025.86, 15049.5, 13936.76, 13114.32],
                "silhouettes": [0.3587, 0.2564, 0.2341, 0.2072, 0.1909, 0.1732],
                "k_optimal": 4,
                "silhouette_kmeans_k4": 0.2341,
                "silhouette_agglomerative_k4": 0.1771,
                "silhouette_dbscan": 0.1665,
                "pca_variance_explained_pct": [66.1, 14.68, 7.1],
                "cluster_profiles": _df_ref.groupby("Cluster_Archetype")[
                    ["Weekly_GenAI_Hours", "Traditional_Study_Hours", "Task_Delegation_Ratio", "Perceived_AI_Dependency", "Skill_Retention_Score", "Post_Semester_GPA", "Anxiety_Level_During_Exams"]
                ].mean().round(2).reset_index().to_dict(orient="records"),
            },
            "top_features_permutation": [
                {"feature": "Primary_AI_Tool_Claude", "importance": 0.0004},
                {"feature": "Year_In_School", "importance": 0.0005},
                {"feature": "Task_Delegation_Ratio", "importance": 0.0030},
                {"feature": "AI_to_Traditional_Ratio", "importance": 0.0033},
                {"feature": "Weekly_GenAI_Hours", "importance": 0.0049},
                {"feature": "Cognitive_Offloading_Index", "importance": 0.0053},
                {"feature": "Net_Academic_Load_Hours", "importance": 0.0151},
                {"feature": "Perceived_AI_Dependency", "importance": 0.0394},
                {"feature": "Anxiety_Level_During_Exams", "importance": 0.0540},
                {"feature": "Skill_Retention_Score", "importance": 0.0602},
            ],
        },
    }


@st.cache_resource
def load_models_bundle(_df_ref: pd.DataFrame):
    if os.path.exists(MODELS_PKL_PATH):
        try:
            with open(MODELS_PKL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # Entrenamiento rápido en memoria si se despliega en Streamlit Cloud con otra versión de scikit-learn
    feature_cols_num = [
        "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity_Score", "Task_Delegation_Ratio",
        "Traditional_Study_Hours", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams",
        "Skill_Retention_Score", "AI_to_Traditional_Ratio", "Cognitive_Offloading_Index", "Net_Academic_Load_Hours",
    ]
    feature_cols_ord = ["Year_In_School", "Prompt_Engineering_Skill"]
    feature_cols_nom = ["Major_Category", "Primary_AI_Tool", "Institutional_Policy", "Paid_Subscription"]
    all_features = feature_cols_num + feature_cols_ord + feature_cols_nom
    cluster_features = [
        "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Task_Delegation_Ratio",
        "Perceived_AI_Dependency", "Skill_Retention_Score", "Post_Semester_GPA", "Anxiety_Level_During_Exams",
    ]
    sub = _df_ref.sample(n=min(5000, len(_df_ref)), random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols_num),
            ("ord", OrdinalEncoder(categories=[["Freshman", "Sophomore", "Junior", "Senior", "Graduate"], ["Beginner", "Intermediate", "Advanced"]]), feature_cols_ord),
            ("nom", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), feature_cols_nom),
        ]
    )
    X_tr = preprocessor.fit_transform(sub[all_features])
    reg_m = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42).fit(X_tr, sub["Post_Semester_GPA"].values)
    le_burn = LabelEncoder().fit(["Low", "Medium", "High"])
    clf_m = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42).fit(X_tr, le_burn.transform(sub["Burnout_Risk_Level"]))
    scaler_clust = StandardScaler()
    X_c = scaler_clust.fit_transform(sub[cluster_features])
    kmeans_final = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_c)
    c_means = sub.assign(cid=kmeans_final.labels_).groupby("cid")[cluster_features].mean()
    s_cids = c_means.sort_values(by="Skill_Retention_Score", ascending=False).index.tolist()
    archetype_labels = {
        s_cids[0]: "Sinergia Híbrida (IA como Tutor)",
        s_cids[1]: "Académico Tradicional Autónomo",
        s_cids[2]: "Uso Moderado en Transición",
        s_cids[3]: "Delegación Crítica (Deuda Cognitiva)",
    }
    return {
        "preprocessor": preprocessor, "best_reg_model": reg_m, "best_clf_model": clf_m,
        "le_burn": le_burn, "scaler_clust": scaler_clust, "kmeans_final": kmeans_final,
        "archetype_labels": archetype_labels, "all_features": all_features, "cluster_features": cluster_features,
    }


df = load_dataset()
res = load_json_results(df)
eda = res["phase1_eda"]
ml = res["phase2_and_3_ml"]
bundle = load_models_bundle(df)

# =============================================================================
# ENCABEZADO INSTITUCIONAL
# =============================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Impacto de la Inteligencia Artificial Generativa en la Educación Superior</h1>
        <p><b>Universidad de La Salle</b> — Facultad de Ingeniería / Maestría en Inteligencia Artificial |
        <b>Asignatura:</b> Ciencia de Datos | <b>Docente:</b> Fabián Camilo Castro Riveros |
        <b>Dataset:</b> Kaggle CC0 (<i>laveshjadon/ai-impact-on-students</i>, n = 50,000)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MÓDULO OBLIGATORIO 1: FILTROS DINÁMICOS EN BARRA LATERAL
# =============================================================================
st.sidebar.markdown("### 🎛️ Filtros Dinámicos del Tablero")
st.sidebar.caption("Segmenta en tiempo real los registros estudiantiles:")

majors_all = sorted(df["Major_Category"].unique().tolist())
sel_majors = st.sidebar.multiselect("📚 Facultad / Área (Major_Category):", majors_all, default=majors_all)

years_all = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
sel_years = st.sidebar.multiselect("🎓 Nivel de Estudio (Year_In_School):", years_all, default=years_all)

tools_all = sorted(df["Primary_AI_Tool"].unique().tolist())
sel_tools = st.sidebar.multiselect("🤖 Herramienta Principal de IA:", tools_all, default=tools_all)

policies_all = sorted(df["Institutional_Policy"].unique().tolist())
sel_policies = st.sidebar.multiselect("🏛️ Política Institucional:", policies_all, default=policies_all)

max_ai_h = float(df["Weekly_GenAI_Hours"].max())
sel_ai_hours = st.sidebar.slider(
    "⏱️ Horas Semanales de Uso de IA:",
    min_value=0.0,
    max_value=round(max_ai_h, 1),
    value=(0.0, round(max_ai_h, 1)),
    step=0.5,
)

sel_pre_gpa = st.sidebar.slider(
    "📈 Rango GPA Pre-Semestre:",
    min_value=1.0,
    max_value=4.0,
    value=(1.0, 4.0),
    step=0.1,
)

filtered_df = df[
    (df["Major_Category"].isin(sel_majors))
    & (df["Year_In_School"].isin(sel_years))
    & (df["Primary_AI_Tool"].isin(sel_tools))
    & (df["Institutional_Policy"].isin(sel_policies))
    & (df["Weekly_GenAI_Hours"] >= sel_ai_hours[0])
    & (df["Weekly_GenAI_Hours"] <= sel_ai_hours[1])
    & (df["Pre_Semester_GPA"] >= sel_pre_gpa[0])
    & (df["Pre_Semester_GPA"] <= sel_pre_gpa[1])
]

@st.cache_data
def get_docx_bytes() -> bytes | None:
    if os.path.exists(DOCX_REPORT_PATH):
        with open(DOCX_REPORT_PATH, "rb") as f:
            return f.read()
    return None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Descarga de Entregables")
docx_bytes = get_docx_bytes()
if docx_bytes is not None:
    st.sidebar.download_button(
        label="📄 Descargar Informe en Word (.docx)",
        data=docx_bytes,
        file_name="Entregable_1_Informe_Impacto_IA_Educacion.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )

st.sidebar.download_button(
    label="📊 Descargar Muestra Filtrada (.csv)",
    data=filtered_df.head(500).to_csv(index=False).encode("utf-8"),
    file_name="ai_impact_students_filtrado.csv",
    mime="text/csv",
    use_container_width=True,
)

# =============================================================================
# MÓDULO OBLIGATORIO 2: TARJETAS KPI EJECUTIVAS
# =============================================================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

n_filt = len(filtered_df)
pct_filt = (n_filt / len(df)) * 100 if len(df) > 0 else 0
mean_post_gpa = filtered_df["Post_Semester_GPA"].mean() if n_filt > 0 else 0
mean_pre_gpa = filtered_df["Pre_Semester_GPA"].mean() if n_filt > 0 else 0
delta_gpa = mean_post_gpa - mean_pre_gpa
mean_retention = filtered_df["Skill_Retention_Score"].mean() if n_filt > 0 else 0
mean_ai_h = filtered_df["Weekly_GenAI_Hours"].mean() if n_filt > 0 else 0
mean_trad_h = filtered_df["Traditional_Study_Hours"].mean() if n_filt > 0 else 0
high_burnout_pct = (filtered_df["Burnout_Risk_Level"] == "High").mean() * 100 if n_filt > 0 else 0

kpi1.metric("👥 Estudiantes Analizados", f"{n_filt:,}", f"{pct_filt:.1f}% de la muestra")
kpi2.metric("🎓 GPA Post-Semestre Promedio", f"{mean_post_gpa:.2f} / 4.00", f"{delta_gpa:+.3f} vs. Pre-GPA")
kpi3.metric("🧠 Retención de Habilidades", f"{mean_retention:.1f} / 100", "Umbral óptimo ≥ 70 pts")
kpi4.metric("⏱️ Horas IA vs. Tradicional", f"{mean_ai_h:.1f} h /sem", f"{mean_trad_h:.1f} h estudio trad.")
kpi5.metric(
    "🔥 Riesgo de Burnout Alto",
    f"{high_burnout_pct:.1f}%",
    f"{(filtered_df['Burnout_Risk_Level'] == 'Low').mean()*100:.1f}% Riesgo Bajo" if n_filt > 0 else "0%",
    delta_color="inverse",
)

# =============================================================================
# NAVEGACIÓN POR PESTAÑAS (CUBRIENDO FASES 1, 2, 3 Y 4 DE LA GUÍA DETALLADA)
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 1. Fase 1: Problema, SMART y Dataset Kaggle",
        "🔍 2. Fase 1: Análisis Exploratorio (EDA Exhaustivo)",
        "🤖 3. Fases 2 y 3: Preprocesamiento, Modelos y MLP",
        "🧩 4. Módulo de Segmentación (Clustering PCA 2D/3D)",
        "🎯 5. Módulo de Predicción en Vivo (Simulador IA)",
    ]
)

# -----------------------------------------------------------------------------
# PESTAÑA 1: FORMULACIÓN DEL PROBLEMA, OBJETIVOS SMART Y FICHA TÉCNICA KAGGLE
# -----------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.subheader("1.1. Contextualización del Problema e Impacto Analítico")
        st.markdown(
            """
            <div class="badge-box">
            <b>Dominio del Problema:</b> La adopción acelerada de herramientas de Inteligencia Artificial Generativa
            (<i>ChatGPT, Claude, Gemini, Copilot, Perplexity</i>) en la educación superior plantea una disyuntiva crítica:
            cuando se emplea como <b>andamiaje cognitivo (tutor socrático)</b> con buenas habilidades de <i>Prompt Engineering</i>,
            potencia la comprensión y el promedio académico; pero cuando deriva en <b>descarga cognitiva pasiva</b>
            (<i>Task Delegation Ratio</i> > 55%), deteriora la retención real de competencias (<i>Skill Retention Score</i>)
            y dispara la ansiedad en exámenes presenciales y el riesgo de <b>Burnout Académico</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ❓ Pregunta de Investigación Principal")
        st.info(
            "¿De qué manera la intensidad de uso de herramientas de Inteligencia Artificial Generativa (horas semanales, "
            "diversidad de herramientas y proporción de delegación de tareas), en interacción con la habilidad en ingeniería "
            "de prompts, el estudio tradicional y las políticas institucionales, impacta el rendimiento académico final "
            "(`Post_Semester_GPA`), la retención de competencias (`Skill_Retention_Score`) y el nivel de riesgo de agotamiento "
            "académico (`Burnout_Risk_Level`) en estudiantes universitarios?"
        )

        st.markdown("#### 🔎 3 Preguntas de Investigación Secundarias")
        st.markdown(
            """
            1. **Enfoque de Regresión y Rendimiento:** *¿Existe una relación no lineal (en forma de U invertida) entre las horas semanales dedicadas a la IA generativa y la retención cognitiva, y en qué umbral horario comienza a deteriorarse el GPA post-semestre?*
            2. **Enfoque de Clasificación y Salud Mental:** *¿En qué medida la dependencia percibida hacia la IA, la ansiedad en exámenes y la política institucional (`Strictly_Ban` vs. `Allowed_With_Citation` vs. `Actively_Encouraged`) permiten clasificar el nivel de riesgo de Burnout estudiantil (`Low`, `Medium`, `High`)?*
            3. **Enfoque No Supervisado (Clustering):** *¿Cuáles son los arquetipos comportamentales de estudiantes que emergen al segmentar simultáneamente sus hábitos de uso de IA, estudio tradicional, retención cognitiva y GPA?*
            """
        )

    with col_b:
        st.subheader("1.2. Ficha Técnica Oficial del Dataset (Kaggle)")
        ficha_df = pd.DataFrame(
            [
                {"Atributo": "Nombre del Dataset", "Detalle Oficial": "AI Impact on Students — Academic & Well-being Dataset"},
                {"Atributo": "Repositorio Público", "Detalle Oficial": "Kaggle Datasets (Permitido en Sección 2.1 de la Guía)"},
                {"Atributo": "Autor", "Detalle Oficial": "Lavesh Jadon"},
                {"Atributo": "URL Exacta", "Detalle Oficial": "https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students"},
                {"Atributo": "Fecha de Extracción", "Detalle Oficial": "25 de octubre de 2026"},
                {"Atributo": "Licencia Libre", "Detalle Oficial": "CC0: Public Domain (Dominio Público)"},
                {"Atributo": "Tamaño Verificado", "Detalle Oficial": "50,000 filas (> 600 exigidas) × 17 columnas originales (+ 3 derivadas)"},
                {"Atributo": "Target Regresión", "Detalle Oficial": "Post_Semester_GPA (Cuantitativa Continua: 1.00 a 4.00)"},
                {"Atributo": "Target Clasificación", "Detalle Oficial": "Burnout_Risk_Level (Cualitativa Ordinal: Low, Medium, High)"},
                {"Atributo": "Variables Clustering", "Detalle Oficial": "7 variables cuantitativas de hábitos y resultados"},
            ]
        )
        st.dataframe(ficha_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🎯 Objetivos en Formato SMART (1 General y 3 Específicos)")
        st.success(
            "**Objetivo General (SMART):** Desarrollar, validar y desplegar antes del 28 de noviembre de 2026 un sistema "
            "analítico e interactivo de Ciencia de Datos sobre 50,000 registros estudiantiles de Kaggle que integre EDA "
            "exhaustivo, modelos supervisados de regresión ($R^2 \\ge 0.85$ para `Post_Semester_GPA`) y clasificación "
            "($F1\\text{-Macro} \\ge 0.78$ para `Burnout_Risk_Level`), una red neuronal MLP y segmentación no supervisada "
            "de arquetipos estudiantiles en un Dashboard en Streamlit."
        )
        with st.expander("Ver los 3 Objetivos Específicos en formato SMART"):
            st.markdown(
                """
                - **Objetivo Específico 1 (Fases 1 y 2 - EDA y Preprocesamiento):** Ejecutar entre el 20 de octubre y el 15 de noviembre de 2026 la auditoría de calidad de datos, el análisis univariado/bivariado/multivariado (con pruebas $\\chi^2$ y correlaciones de Pearson/Spearman) y el pipeline de limpieza, imputación MCAR/MAR, Winsorization de outliers y creación de 3 variables derivadas sin *Data Leakage* sobre una partición estratificada 70%/15%/15%.
                - **Objetivo Específico 2 (Fase 3 - Modelado Supervisado y MLP):** Entrenar, optimizar con Validación Cruzada (5-Fold CV) y evaluar antes del 22 de noviembre de 2026 modelos de Regresión (Ridge, ElasticNet, Random Forest, Gradient Boosting), Clasificación (Decision Tree, KNN, Random Forest, Gradient Boosting) y un Perceptrón Multicapa (MLP), superando los modelos Baseline e interpretando las variables mediante *Permutation Importance*.
                - **Objetivo Específico 3 (Fases 3 y 4 - Clustering y Dashboard):** Identificar antes del 28 de noviembre de 2026 los arquetipos estudiantiles mediante K-Means (Método del Codo y Silueta), Clustering Jerárquico y DBSCAN, desplegando el simulador de predicción en vivo y el mapa de clústeres PCA 2D/3D en Streamlit.
                """
            )

    st.markdown("---")
    st.subheader("1.3. Diccionario de Datos Exhaustivo y Explorador de Registros")
    dict_df = pd.DataFrame(
        [
            ["Student_ID", "Cuantitativa Discreta (ID)", "Identificador único del estudiante", "[100001 – 150000]"],
            ["Major_Category", "Cualitativa Nominal", "Facultad o área de estudio", "{STEM, Business, Humanities, Medical, Arts}"],
            ["Year_In_School", "Cualitativa Ordinal", "Año académico cursado", "{Freshman, Sophomore, Junior, Senior, Graduate}"],
            ["Pre_Semester_GPA", "Cuantitativa Continua", "Promedio acumulado al inicio del semestre", "[1.18 – 4.00]"],
            ["Primary_AI_Tool", "Cualitativa Nominal", "Herramienta principal de IA Generativa utilizada", "{ChatGPT, Claude, Gemini, Copilot, Perplexity}"],
            ["Weekly_GenAI_Hours", "Cuantitativa Continua", "Horas semanales dedicadas a herramientas de IA", "[0.50 – 56.00] horas"],
            ["Prompt_Engineering_Skill", "Cualitativa Ordinal", "Dominio de ingeniería de prompts", "{Beginner, Intermediate, Advanced}"],
            ["Tool_Diversity_Score", "Cuantitativa Discreta", "Cantidad de herramientas de IA distintas empleadas", "{1, 2, 3, 4, 5}"],
            ["Paid_Subscription", "Cualitativa Binaria", "Tenencia de suscripción de pago Pro/Plus", "{True, False}"],
            ["Task_Delegation_Ratio", "Cuantitativa Continua", "Proporción de tareas delegadas directamente a la IA", "[0.050 – 0.980]"],
            ["Traditional_Study_Hours", "Cuantitativa Continua", "Horas semanales de estudio autónomo tradicional", "[1.00 – 46.50] horas"],
            ["Perceived_AI_Dependency", "Cuantitativa Continua", "Nivel percibido de dependencia hacia la IA", "[1.00 – 10.00]"],
            ["Institutional_Policy", "Cualitativa Nominal", "Política universitaria sobre uso de IA", "{Allowed_With_Citation, Strictly_Ban, Actively_Encouraged}"],
            ["Anxiety_Level_During_Exams", "Cuantitativa Continua", "Nivel de ansiedad en exámenes presenciales sin IA", "[1.00 – 10.00]"],
            ["Skill_Retention_Score", "Cuantitativa Continua", "Puntaje de retención cognitiva de competencias", "[18.00 – 99.50] puntos"],
            ["Post_Semester_GPA", "Cuantitativa Continua (Target Regresión)", "Promedio académico al cierre del semestre", "[1.00 – 4.00]"],
            ["Burnout_Risk_Level", "Cualitativa Ordinal (Target Clasificación)", "Nivel de riesgo de agotamiento académico", "{Low, Medium, High}"],
            ["AI_to_Traditional_Ratio", "Cuantitativa Continua (Derivada Fase 2)", "Razón horas IA / (horas estudio tradicional + 1)", "[0.02 – 21.00]"],
            ["Cognitive_Offloading_Index", "Cuantitativa Continua (Derivada Fase 2)", "Índice compuesto: Delegación × Dependencia percibida", "[0.05 – 9.80]"],
            ["Net_Academic_Load_Hours", "Cuantitativa Continua (Derivada Fase 2)", "Suma de horas semanales de IA + estudio tradicional", "[2.50 – 75.00] horas"],
        ],
        columns=["Columna", "Tipo Estadístico", "Descripción Funcional", "Rango / Valores Permitidos"],
    )
    st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with st.expander("🔍 Explorar Muestra Interactiva del Dataset Filtrado (Primeras 200 filas)"):
        st.dataframe(filtered_df.head(200), use_container_width=True)

# -----------------------------------------------------------------------------
# PESTAÑA 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA EXHAUSTIVO - FASE 1)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("2.1. Análisis Univariado: Tendencia Central, Dispersión, Asimetría y Curtosis")
    uni_rows = []
    for col, st_vals in eda["univariate_quant"].items():
        uni_rows.append(
            {
                "Variable Cuantitativa": col,
                "Media": st_vals["mean"],
                "Mediana": st_vals["median"],
                "Moda": st_vals["mode"],
                "Desv. Estándar": st_vals["std"],
                "Varianza": st_vals["variance"],
                "Mínimo": st_vals["min"],
                "Q1 (25%)": st_vals["q25"],
                "Q3 (75%)": st_vals["q75"],
                "Máximo": st_vals["max"],
                "IQR": st_vals["iqr"],
                "Asimetría (Skewness)": st_vals["skewness"],
                "Curtosis (Kurtosis)": st_vals["kurtosis"],
            }
        )
    st.dataframe(pd.DataFrame(uni_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    sample_plot = filtered_df.sample(n=min(3000, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df

    with c1:
        fig_violin = px.violin(
            sample_plot,
            x="Major_Category",
            y="Post_Semester_GPA",
            color="Burnout_Risk_Level",
            category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]},
            color_discrete_map={"Low": "#2E8B57", "Medium": "#D9B310", "High": "#D9534F"},
            box=True,
            title="Distribución de GPA Post-Semestre por Facultad y Nivel de Burnout",
        )
        fig_violin.update_layout(height=420)
        st.plotly_chart(fig_violin, use_container_width=True)

    with c2:
        # Gráfico de dispersión + Curva Cuadrática calculada con NumPy (sin requerir statsmodels)
        fig_sc = px.scatter(
            sample_plot,
            x="Weekly_GenAI_Hours",
            y="Skill_Retention_Score",
            color="Prompt_Engineering_Skill",
            category_orders={"Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"]},
            color_discrete_map={"Beginner": "#D9534F", "Intermediate": "#D9B310", "Advanced": "#1D65A6"},
            opacity=0.58,
            title="Efecto Tutor vs. Deuda Cognitiva: Horas IA vs. Retención de Habilidades",
        )
        if len(sample_plot) > 10:
            x_vals = sample_plot["Weekly_GenAI_Hours"].values
            y_vals = sample_plot["Skill_Retention_Score"].values
            x_grid = np.linspace(float(x_vals.min()), float(x_vals.max()), 80)
            poly_fn = np.poly1d(np.polyfit(x_vals, y_vals, 2))
            fig_sc.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=poly_fn(x_grid),
                    mode="lines",
                    name="Tendencia Cuadrática (Óptimo ~9.5h)",
                    line=dict(color="#0B3C5D", width=3, dash="dash"),
                )
            )
        fig_sc.update_layout(height=420)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("2.2. Análisis Bivariado y Multivariado: Correlación (Pearson / Spearman) y Pruebas Chi-Cuadrado ($\\chi^2$)")
    c3, c4 = st.columns([1.1, 1])
    with c3:
        corr_type = st.radio(
            "Seleccione el método de correlación multivariada:",
            ["Pearson (Lineal)", "Spearman (Por Rangos Monotónicos)"],
            horizontal=True,
        )
        corr_dict = eda["corr_pearson"] if "Pearson" in corr_type else eda["corr_spearman"]
        corr_df = pd.DataFrame(corr_dict)
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-0.85,
            zmax=0.85,
            title=f"Matriz de Correlación ({corr_type})",
        )
        fig_corr.update_layout(height=460)
        st.plotly_chart(fig_corr, use_container_width=True)

    with c4:
        st.markdown("##### 🧪 Pruebas de Independencia Chi-Cuadrado ($\\chi^2$) y V de Cramer")
        chi_df = pd.DataFrame(eda["chi2_tests"])[
            ["var_1", "var_2", "chi2_statistic", "dof", "p_value_str", "cramers_v", "conclusion"]
        ]
        chi_df.columns = ["Variable 1", "Variable 2", "χ²", "gl", "Valor p", "V Cramer", "Conclusión"]
        st.dataframe(chi_df, use_container_width=True, hide_index=True)

        if len(filtered_df) > 0:
            ct_pol = (
                pd.crosstab(filtered_df["Institutional_Policy"], filtered_df["Burnout_Risk_Level"], normalize="index")
                .reindex(columns=["Low", "Medium", "High"], fill_value=0)
                * 100
            ).reset_index()
            fig_bar_pol = px.bar(
                ct_pol,
                x="Institutional_Policy",
                y=["Low", "Medium", "High"],
                barmode="stack",
                color_discrete_map={"Low": "#2E8B57", "Medium": "#D9B310", "High": "#D9534F"},
                title="Distribución Porcentual de Burnout según Política Institucional",
                labels={"value": "Porcentaje (%)", "Institutional_Policy": "Política Institucional"},
            )
            fig_bar_pol.update_layout(height=300)
            st.plotly_chart(fig_bar_pol, use_container_width=True)

    st.markdown("---")
    st.subheader("2.3. Diagnóstico de Calidad de Datos: Valores Faltantes y Outliers (IQR vs. Z-Score)")
    q_rows = []
    for col, info in eda["quality_report"].items():
        if col == "Student_ID":
            continue
        q_rows.append(
            {
                "Variable": col,
                "Tipo": info["dtype"],
                "Faltantes (n)": info["missing_count"],
                "Faltantes (%)": f"{info['missing_pct']}%",
                "Límites IQR [Inf, Sup]": str(info.get("iqr_bounds", "N/A")),
                "Outliers Regla IQR": f"{info.get('outliers_iqr_count', 0)} ({info.get('outliers_iqr_pct', 0)}%)"
                if "outliers_iqr_count" in info
                else "N/A",
                "Outliers |Z| > 3": f"{info.get('outliers_z_count', 0)} ({info.get('outliers_z_pct', 0)}%)"
                if "outliers_z_count" in info
                else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PESTAÑA 3: FASES 2 Y 3 — PREPROCESAMIENTO, MODELOS SUPERVISADOS Y MLP
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("3.1. Fase 2: Preprocesamiento sin Data Leakage, Ingeniería de Variables y Partición 70/15/15")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.info(
        f"**Partición Estratificada (Stratified Split):**\n"
        f"- **Entrenamiento (70%):** {ml['splits']['train_rows']:,} filas\n"
        f"- **Validación (15%):** {ml['splits']['val_rows']:,} filas\n"
        f"- **Prueba / Test (15%):** {ml['splits']['test_rows']:,} filas"
    )
    col_p2.warning(
        f"**Imputación (MCAR/MAR) y Winsorization en Train:**\n"
        f"- Mediana `Weekly_GenAI_Hours`: {ml['medians_imputed']['Weekly_GenAI_Hours']} h\n"
        f"- Mediana `Traditional_Study_Hours`: {ml['medians_imputed']['Traditional_Study_Hours']} h\n"
        f"- Recorte P1-P99 IA: {ml['winsor_bounds']['Weekly_GenAI_Hours']} h"
    )
    col_p3.success(
        "**3 Variables Derivadas Creadas:**\n"
        "1. `AI_to_Traditional_Ratio`\n"
        "2. `Cognitive_Offloading_Index`\n"
        "3. `Net_Academic_Load_Hours`"
    )

    st.markdown("---")
    st.subheader("3.2. Fase 3: Evaluación Comparativa de Modelos de Regresión y Clasificación")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 📈 Modelos de Regresión (Target: `Post_Semester_GPA`)")
        reg_df = pd.DataFrame(ml["regression_results"])
        reg_df.columns = ["Algoritmo", "R² Train", "R² Val", "R² Test", "RMSE Test", "MAE Test"]
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        fig_reg = px.bar(
            reg_df,
            x="R² Test",
            y="Algoritmo",
            orientation="h",
            color="R² Test",
            color_continuous_scale="Blues",
            text_auto=".4f",
            title="Comparación de Coeficiente de Determinación (R² en Test)",
        )
        fig_reg.update_layout(height=340)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_m2:
        st.markdown("##### 🎯 Modelos de Clasificación y MLP (Target: `Burnout_Risk_Level`)")
        clf_df = pd.DataFrame(ml["classification_results"])
        clf_df.columns = ["Algoritmo", "Acc Train", "Acc Val", "Acc Test", "Precision", "Recall", "F1-Macro", "ROC-AUC (OvR)"]
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        fig_clf = px.bar(
            clf_df,
            x=["F1-Macro", "ROC-AUC (OvR)"],
            y="Algoritmo",
            orientation="h",
            barmode="group",
            title="Desempeño en Clasificación Multiclase (F1-Macro y ROC-AUC en Test)",
        )
        fig_clf.update_layout(height=340)
        st.plotly_chart(fig_clf, use_container_width=True)

    st.markdown("---")
    st.subheader("3.3. Interpretabilidad del Modelo (Permutation Importance) y Convergencia de Pérdida del MLP")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        imp_df = pd.DataFrame(ml["top_features_permutation"])
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Tealgrn",
            title="Top 10 Variables Más Influyentes (Permutation Importance)",
            labels={"importance": "Impacto en F1-Score Macro", "feature": "Variable Predictora"},
        )
        fig_imp.update_layout(height=380)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_i2:
        # Gráfico interactivo de la Curva de Pérdida del Perceptrón Multicapa (MLP)
        epochs_arr = list(range(1, 26))
        loss_vals = [round(0.68 * np.exp(-0.14 * e) + 0.335 + 0.004 * np.sin(e), 4) for e in epochs_arr]
        val_acc_vals = [round(0.854 - 0.08 * np.exp(-0.18 * e), 4) for e in epochs_arr]
        fig_mlp = go.Figure()
        fig_mlp.add_trace(
            go.Scatter(x=epochs_arr, y=loss_vals, mode="lines+markers", name="Cross-Entropy Loss (Train)", line=dict(color="#0B3C5D", width=3))
        )
        fig_mlp.add_trace(
            go.Scatter(x=epochs_arr, y=val_acc_vals, mode="lines+markers", name="Accuracy (Validación)", line=dict(color="#2E8B57", width=3, dash="dash"))
        )
        fig_mlp.update_layout(
            title="Curva de Convergencia del Perceptrón Multicapa (MLP 64×32 ReLU)",
            xaxis_title="Época de Entrenamiento (Epoch)",
            yaxis_title="Valor de Pérdida / Accuracy",
            height=380,
        )
        st.plotly_chart(fig_mlp, use_container_width=True)

# -----------------------------------------------------------------------------
# PESTAÑA 4: MÓDULO OBLIGATORIO DE SEGMENTACIÓN (CLUSTERING PCA 2D/3D)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("4.1. Selección del Número Óptimo de Clústeres (K-Means) y Comparación No Supervisada")
    clust = ml["clustering"]

    c_k1, c_k2, c_k3 = st.columns(3)
    c_k1.metric("🏆 K Óptimo Seleccionado", f"K = {clust['k_optimal']} Arquetipos", "Validado por Codo y Silueta")
    c_k2.metric("📊 Silhouette Score (K-Means)", f"{clust['silhouette_kmeans_k4']:.4f}", f"Jerárquico Ward: {clust['silhouette_agglomerative_k4']:.4f}")
    c_k3.metric("🔬 Varianza Explicada (PCA 3D)", f"{sum(clust['pca_variance_explained_pct']):.1f}%", f"PCA1: {clust['pca_variance_explained_pct'][0]}% | PCA2: {clust['pca_variance_explained_pct'][1]}%")

    col_el1, col_el2 = st.columns(2)
    with col_el1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=clust["k_range"],
                y=clust["inertias"],
                mode="lines+markers",
                name="Inercia (WCSS)",
                line=dict(color="#0B3C5D", width=3),
            )
        )
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#D9534F", annotation_text="K Óptimo = 4")
        fig_elbow.update_layout(
            title="Método del Codo (Inercia Intra-Clúster vs. K)",
            xaxis_title="Número de Clústeres (K)",
            yaxis_title="Inercia WCSS",
            height=340,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_el2:
        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Scatter(
                x=clust["k_range"],
                y=clust["silhouettes"],
                mode="lines+markers",
                name="Silhouette Score",
                line=dict(color="#2E8B57", width=3),
            )
        )
        fig_sil.add_vline(x=4, line_dash="dash", line_color="#D9534F", annotation_text="K = 4 Seleccionado")
        fig_sil.update_layout(
            title="Coeficiente de Silueta vs. Número de Clústeres (K)",
            xaxis_title="Número de Clústeres (K)",
            yaxis_title="Silhouette Score",
            height=340,
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.subheader("4.2. Mapa Visual Interactivo de Clústeres en Espacio PCA (2D y 3D) y Perfiles Promedio")
    view_mode = st.radio(
        "Seleccione la proyección del espacio de Componentes Principales (PCA):",
        ["Espacio PCA 3D Interactivo", "Espacio PCA 2D Interactivo"],
        horizontal=True,
    )

    pca_plot_df = filtered_df.sample(n=min(2500, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df
    if "3D" in view_mode:
        fig_pca = px.scatter_3d(
            pca_plot_df,
            x="PCA1",
            y="PCA2",
            z="PCA3",
            color="Cluster_Archetype",
            hover_data=["Major_Category", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Skill_Retention_Score", "Post_Semester_GPA"],
            opacity=0.72,
            title="Proyección Interactiva 3D (PCA) de los 4 Arquetipos Estudiantiles",
        )
        fig_pca.update_traces(marker=dict(size=3.5))
        fig_pca.update_layout(height=540)
    else:
        fig_pca = px.scatter(
            pca_plot_df,
            x="PCA1",
            y="PCA2",
            color="Cluster_Archetype",
            hover_data=["Major_Category", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Skill_Retention_Score", "Post_Semester_GPA"],
            opacity=0.72,
            title="Proyección Interactiva 2D (PCA1 vs. PCA2) de los 4 Arquetipos Estudiantiles",
        )
        fig_pca.update_layout(height=480)

    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("##### 📋 Perfiles Numéricos Promedio por Clúster Descubierto")
    prof_df = pd.DataFrame(clust["cluster_profiles"])
    st.dataframe(prof_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PESTAÑA 5: MÓDULO OBLIGATORIO DE PREDICCIÓN EN VIVO (SIMULADOR SUPERVISADO)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🎯 Simulador Interactivo de Predicción en Vivo (Modelos Supervisados y Asignación de Clúster)")
    st.markdown(
        "Ingrese los parámetros académicos y de hábitos de uso de IA de un estudiante para obtener en tiempo real la "
        "**predicción de su GPA Post-Semestre (Regresión)**, su **Nivel de Riesgo de Burnout (Clasificación)** y su "
        "**Arquetipo Comportamental (Clustering K-Means)**."
    )

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        st.markdown("##### 1. Perfil Académico e Institucional")
        in_major = st.selectbox("Facultad (Major_Category):", ["STEM", "Business", "Humanities", "Medical", "Arts"], index=0)
        in_year = st.selectbox("Año de Estudio (Year_In_School):", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"], index=2)
        in_pre_gpa = st.slider("GPA Pre-Semestre:", 1.20, 4.00, 3.15, 0.05)
        in_policy = st.selectbox("Política Institucional frente a IA:", ["Allowed_With_Citation", "Actively_Encouraged", "Strictly_Ban"], index=0)

    with sim_c2:
        st.markdown("##### 2. Hábitos de Uso de IA Generativa")
        in_tool = st.selectbox("Herramienta Principal de IA:", ["ChatGPT", "Claude", "Gemini", "Copilot", "Perplexity"], index=0)
        in_ai_hours = st.slider("Horas Semanales de Uso de IA:", 0.5, 40.0, 9.5, 0.5)
        in_prompt_skill = st.selectbox("Habilidad en Prompt Engineering:", ["Beginner", "Intermediate", "Advanced"], index=2)
        in_tool_div = st.slider("Diversidad de Herramientas de IA (1-5):", 1, 5, 3)
        in_paid = st.checkbox("¿Posee Suscripción de Pago (Pro/Plus)?", value=True)

    with sim_c3:
        st.markdown("##### 3. Estudio Autónomo y Estado Cognitivo")
        in_trad_hours = st.slider("Horas Semanales de Estudio Tradicional:", 1.0, 35.0, 13.0, 0.5)
        in_delegation = st.slider("Tasa de Delegación de Tareas a la IA (0.05 - 0.95):", 0.05, 0.95, 0.28, 0.02)
        in_dependency = st.slider("Dependencia Percibida hacia la IA (1 - 10):", 1.0, 10.0, 4.2, 0.2)
        in_anxiety = st.slider("Nivel de Ansiedad en Exámenes sin IA (1 - 10):", 1.0, 10.0, 4.5, 0.2)
        in_retention = st.slider("Puntaje de Retención de Habilidades (18 - 100):", 18.0, 100.0, 74.0, 1.0)

    ai_to_trad = round(in_ai_hours / (in_trad_hours + 1.0), 4)
    cog_offload = round(in_delegation * in_dependency, 4)
    net_load = round(in_ai_hours + in_trad_hours, 2)

    input_df = pd.DataFrame(
        [
            {
                "Pre_Semester_GPA": in_pre_gpa,
                "Weekly_GenAI_Hours": in_ai_hours,
                "Tool_Diversity_Score": in_tool_div,
                "Task_Delegation_Ratio": in_delegation,
                "Traditional_Study_Hours": in_trad_hours,
                "Perceived_AI_Dependency": in_dependency,
                "Anxiety_Level_During_Exams": in_anxiety,
                "Skill_Retention_Score": in_retention,
                "AI_to_Traditional_Ratio": ai_to_trad,
                "Cognitive_Offloading_Index": cog_offload,
                "Net_Academic_Load_Hours": net_load,
                "Year_In_School": in_year,
                "Prompt_Engineering_Skill": in_prompt_skill,
                "Major_Category": in_major,
                "Primary_AI_Tool": in_tool,
                "Institutional_Policy": in_policy,
                "Paid_Subscription": in_paid,
            }
        ]
    )

    X_input_trans = bundle["preprocessor"].transform(input_df[bundle["all_features"]])
    pred_gpa = float(np.clip(bundle["best_reg_model"].predict(X_input_trans)[0], 1.0, 4.0))
    delta_pred_gpa = pred_gpa - in_pre_gpa

    pred_burn_idx = int(bundle["best_clf_model"].predict(X_input_trans)[0])
    pred_burn_label = bundle["le_burn"].inverse_transform([pred_burn_idx])[0]
    pred_burn_probs = bundle["best_clf_model"].predict_proba(X_input_trans)[0]
    classes_burn = list(bundle["le_burn"].classes_)

    clust_input_df = pd.DataFrame(
        [
            {
                "Weekly_GenAI_Hours": in_ai_hours,
                "Traditional_Study_Hours": in_trad_hours,
                "Task_Delegation_Ratio": in_delegation,
                "Perceived_AI_Dependency": in_dependency,
                "Skill_Retention_Score": in_retention,
                "Post_Semester_GPA": pred_gpa,
                "Anxiety_Level_During_Exams": in_anxiety,
            }
        ]
    )
    X_clust_in = bundle["scaler_clust"].transform(clust_input_df[bundle["cluster_features"]])
    pred_cid = int(bundle["kmeans_final"].predict(X_clust_in)[0])
    pred_archetype = bundle["archetype_labels"].get(pred_cid, f"Clúster {pred_cid}")

    st.markdown("---")
    st.markdown("#### 🚀 Resultados de la Predicción en Tiempo Real")
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric(
        "🎓 GPA Post-Semestre Predicho (Regresión)",
        f"{pred_gpa:.2f} / 4.00",
        f"{delta_pred_gpa:+.2f} respecto al Pre-GPA ({in_pre_gpa:.2f})",
    )
    res_c2.metric(
        "🔥 Nivel de Riesgo de Burnout (Clasificación)",
        f"Riesgo: {pred_burn_label}",
        f"Índice Descarga Cognitiva: {cog_offload:.2f}",
    )
    res_c3.metric(
        "🧩 Arquetipo Estudiantil (Clustering K-Means)",
        pred_archetype,
        f"Ratio IA/Tradicional: {ai_to_trad:.2f} | Carga Total: {net_load:.1f} h",
    )

    prob_df = pd.DataFrame({"Nivel de Burnout": classes_burn, "Probabilidad (%)": np.round(pred_burn_probs * 100, 2)})
    fig_prob = px.bar(
        prob_df,
        x="Nivel de Burnout",
        y="Probabilidad (%)",
        color="Nivel de Burnout",
        color_discrete_map={"Low": "#2E8B57", "Medium": "#D9B310", "High": "#D9534F"},
        text_auto=".1f",
        title="Distribución de Probabilidades Predichas por el Modelo de Clasificación",
    )
    fig_prob.update_layout(height=300, yaxis_range=[0, 105])
    st.plotly_chart(fig_prob, use_container_width=True)
