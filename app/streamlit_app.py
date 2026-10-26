"""SkinRisk Fusion: Multimodal Skin Lesion Risk Classification Demo.

Educational research application comparing metadata-only, image-only,
and multimodal fusion models on the ISIC 2024 Challenge dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure repository root is on sys.path for direct streamlit execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import PRIMARY_CATEGORICAL_FEATURES, PRIMARY_NUMERIC_FEATURES
from src.inference.pipeline import (
    THRESHOLDS,
    predict_image,
    predict_metadata,
    predict_multimodal,
)

# Streamlit Page Config
st.set_page_config(
    page_title="SkinRisk Fusion — Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mandatory Non-Clinical Research Disclaimer
DISCLAIMER_TEXT = (
    "⚠️ **FOR EDUCATIONAL RESEARCH ONLY.** This model is an academic research prototype, "
    "is NOT clinically validated, and MUST NOT be used to diagnose skin cancer, assess personal lesions, "
    "or replace professional examination by a qualified dermatologist."
)


def render_warning_banner():
    st.warning(DISCLAIMER_TEXT, icon="⚠️")


@st.cache_data
def load_final_metrics() -> Dict[str, Any]:
    metrics_path = PROJECT_ROOT / "reports" / "phase6_final_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_overview():
    st.title("🔬 SkinRisk Fusion: Multimodal Lesion Risk Model")
    render_warning_banner()

    st.markdown(
        """
        ### Project Objective & Scope
        **SkinRisk Fusion** investigates whether fusing **2D dermoscopic image tensors** with **clinical and 3D geometric lesion metadata**
        improves malignancy risk ranking for skin cancer triage, compared to single-modality baselines.

        The benchmark uses the **ISIC 2024 Challenge dataset** (total-body photography, 401,059 lesions) under extreme class imbalance 
        (~0.096% malignant prevalence).
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "📊 **Modality A: Metadata-Only**\n\n"
            "- LightGBM GBDT classifier\n"
            "- 50 clinical & 3D Lab geometric features\n"
            "- Pre-selected Primary Candidate"
        )
    with col2:
        st.info(
            "🖼️ **Modality B: Image-Only CNN**\n\n"
            "- EfficientNet-B0 (160×160 RGB)\n"
            "- Transfer learning from ImageNet\n"
            "- Single-logit binary head"
        )
    with col3:
        st.info(
            "🧬 **Modality C: Multimodal Fusion**\n\n"
            "- 1,280-d image embedding + 50-d metadata\n"
            "- Dense MLP fusion head (1,330 → 128 → 1)\n"
            "- Evaluated unweighted & weighted"
        )

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            subgraph InputData [Inputs]
                Img[Lesion Image 160x160 RGB]
                Meta[Clinical & 3D Geometric Metadata]
            end

            subgraph FeatureExtraction [Feature Extractors]
                Img --> CNN[EfficientNet-B0 Backbone]
                CNN --> ImgFeat[1,280-d Visual Embedding]
                Meta --> Prep[Leakage-Free ColumnTransformer]
                Prep --> MetaFeat[50-d Preprocessed Metadata]
            end

            subgraph Models [Evaluated Architectures]
                MetaFeat --> LGBM[LightGBM GBDT Baseline]
                ImgFeat --> CNNHead[CNN Classifier Head]
                ImgFeat & MetaFeat --> FusionHead[Concatenation 1,330-d -> MLP Head]
            end

            subgraph Outputs [Outputs & Evaluation]
                LGBM --> Out1[Primary Metadata Score]
                CNNHead --> Out2[Image Score]
                FusionHead --> Out3[Multimodal Score]
                Out1 --> Platt[Validation Platt Calibrator]
                Platt --> OutCal[Calibrated Research Score]
            end
        ```
        """
    )


def render_comparison():
    st.title("📈 Controlled Model Comparison & Final Test Evaluation")
    render_warning_banner()

    st.markdown(
        """
        All candidate models below were trained on the **identical 10,000-lesion matched cohort** 
        (239 malignant, 9,761 benign; prevalence 2.390%) and evaluated on the **identical 80,210-lesion held-out test split** 
        (77 malignant, 80,133 benign; prevalence 0.0960%) across 208 unique patients.
        """
    )

    metrics_data = load_final_metrics()
    if metrics_data and "models" in metrics_data:
        rows = []
        for m in metrics_data["models"]:
            b = m.get("bootstrap", {})
            ap_ci = f"{m['pr_auc']:.5f} [{b.get('ap', {}).get('ci_lower', 0):.4f}, {b.get('ap', {}).get('ci_upper', 0):.4f}]"
            roc_ci = f"{m['roc_auc']:.5f} [{b.get('roc_auc', {}).get('ci_lower', 0):.4f}, {b.get('roc_auc', {}).get('ci_upper', 0):.4f}]"
            cm = f"TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}"
            rows.append({
                "Modality": m["modality"],
                "Model Name": m["name"],
                "Test AP [95% CI]": ap_ci,
                "Test ROC-AUC [95% CI]": roc_ci,
                "Frozen Thresh": f"{m['frozen_threshold']:.4f}",
                "Test Prec": f"{m['precision']:.4f}",
                "Test Recall": f"{m['recall']:.4f}",
                "Test F1": f"{m['f1']:.4f}",
                "Test Confusion Matrix": cm,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.error("Metrics summary file not found at reports/phase6_final_metrics.json.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test Precision-Recall Curves")
        pr_path = PROJECT_ROOT / "reports" / "figures" / "phase6_test_pr_curves.png"
        if pr_path.is_file():
            st.image(str(pr_path), caption="Precision-Recall curves on 80,210 held-out test lesions.")
    with col2:
        st.subheader("Reliability Calibration Curves")
        cal_path = PROJECT_ROOT / "reports" / "figures" / "phase6_calibration_curves.png"
        if cal_path.is_file():
            st.image(str(cal_path), caption="Platt scaling recalibrates predicted probabilities to natural prevalence.")

    st.markdown("---")
    st.subheader("Key Scientific Findings")
    st.markdown(
        """
        1. **Metadata-Only Primary Model Generalization:**  
           The matched LightGBM model generalized effectively from validation (AP = 0.05095) to test (AP = **0.06377**, ROC-AUC = **0.92480**), catching 23 of 77 test malignancies.
        2. **Multimodal Fusion vs. Metadata:**  
           Weighted Multimodal Fusion reached a comparable AP of **0.06471**, but with lower ROC-AUC (0.85176 vs 0.92480) and fewer detected positives (9 vs 23). Standalone LightGBM remains the strongest, most stable single model on this dataset.
        3. **Multimodal Fusion vs. Image CNN:**  
           Fusing clinical metadata with CNN embeddings boosted AP from **0.00972** (CNN alone) to **0.03499** (unweighted fusion) and **0.06471** (weighted fusion), proving that metadata provides massive uplift over image pixels alone.
        4. **Patient-Cluster Uncertainty:**  
           With only 77 test malignant lesions across 208 patients, 95% bootstrap confidence intervals heavily overlap across models.
        """
    )


def render_metadata_inference():
    st.title("📋 Metadata-Only Inference (LightGBM)")
    render_warning_banner()

    st.markdown("Enter lesion clinical and 3D geometric measurements to compute the estimated research risk score.")

    with st.form("metadata_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Patient Demographics & Location")
            sex = st.selectbox("Patient Sex", ["male", "female"])
            age = st.slider("Approximate Age", min_value=10, max_value=90, value=55, step=5)
            anatom = st.selectbox(
                "Anatomical Site",
                ["head/neck", "posterior torso", "anterior torso", "lower extremity", "upper extremity"],
            )
            location_simple = st.selectbox(
                "Simplified Body Location",
                ["Head", "Torso", "Arm", "Leg", "Hand", "Foot", "Neck", "Genital"],
            )
            tile_type = st.selectbox("Camera Modality (Tile Type)", ["3D: XP", "3D: white"])

        with col2:
            st.markdown("#### 3D Geometry & Lesion Size")
            clin_diam = st.number_input("Clinical Long Diameter (mm)", min_value=0.5, max_value=30.0, value=4.5, step=0.1)
            area = st.number_input("Lesion Area (mm²)", min_value=0.5, max_value=200.0, value=15.0, step=0.5)
            perimeter = st.number_input("Lesion Perimeter (mm)", min_value=1.0, max_value=150.0, value=14.0, step=0.5)
            eccentricity = st.slider("Lesion Eccentricity (0=round, 1=elongated)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

        with st.expander("Advanced Lab Colorimetry (Optional, training medians used if left default)"):
            c_a, c_b, c_l = st.columns(3)
            with c_a:
                delta_a = st.number_input("tbp_lv_deltaA", value=-0.2, step=0.1)
            with c_b:
                delta_b = st.number_input("tbp_lv_deltaB", value=-1.5, step=0.1)
            with c_l:
                delta_l = st.number_input("tbp_lv_deltaL", value=-6.0, step=0.5)

        submitted = st.form_submit_button("Compute Metadata Model Score", use_container_width=True)

    if submitted:
        input_data = {
            "sex": sex,
            "age_approx": float(age),
            "anatom_site_general": anatom,
            "tbp_lv_location_simple": location_simple,
            "tbp_tile_type": tile_type,
            "clin_size_long_diam_mm": float(clin_diam),
            "tbp_lv_areaMM2": float(area),
            "tbp_lv_perimeterMM": float(perimeter),
            "tbp_lv_eccentricity": float(eccentricity),
            "tbp_lv_deltaA": float(delta_a),
            "tbp_lv_deltaB": float(delta_b),
            "tbp_lv_deltaL": float(delta_l),
        }

        try:
            res = predict_metadata(input_data)
            st.subheader("Inference Output")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="Raw Model Output Score",
                    value=f"{res['raw_score']:.5f}",
                    help="Uncalibrated score from matched-cohort LightGBM model.",
                )
                st.caption(f"Frozen Operating Threshold: `{res['raw_threshold']:.4f}`")
                if res["is_elevated_risk"]:
                    st.error("Model Risk Flag: Elevated (Score >= Frozen Threshold)")
                else:
                    st.success("Model Risk Flag: Below Threshold")

            with m_col2:
                if "calibrated_score" in res:
                    st.metric(
                        label="Platt-Calibrated Research Risk",
                        value=f"{res['calibrated_score']:.5f}",
                        help="Recalibrated to match real-world 0.096% clinical prevalence.",
                    )
                    st.caption(f"Calibrated Threshold: `{res['calibrated_threshold']:.4f}`")
                    if res.get("is_calibrated_elevated_risk"):
                        st.error("Calibrated Risk Flag: Elevated")
                    else:
                        st.success("Calibrated Risk Flag: Below Threshold")
        except Exception as e:
            st.error(f"Inference error: {e}")


def render_image_inference():
    st.title("🖼️ Image-Only CNN Inference (EfficientNet-B0)")
    render_warning_banner()

    st.markdown(
        """
        Upload a dermoscopic or crop-lesion image (JPEG or PNG).
        The model uses **EfficientNet-B0** ($160 \\times 160$ RGB input) with ImageNet pretrained transfer weights.
        """
    )

    uploaded_file = st.file_uploader("Upload Lesion Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Lesion Image", width=250)

            if st.button("Run Image CNN Inference", use_container_width=True):
                with st.spinner("Processing image on Apple Silicon M3 (MPS)..."):
                    res = predict_image(img)

                st.subheader("Inference Output")
                st.metric(label="CNN Risk Score", value=f"{res['raw_score']:.5f}")
                st.caption(f"Frozen Validation Threshold: `{res['raw_threshold']:.4f}`")
                if res["is_elevated_risk"]:
                    st.error("Model Risk Flag: Elevated (Score >= Frozen Threshold)")
                else:
                    st.success("Model Risk Flag: Below Threshold")
                st.info(
                    "Note: Arbitrary consumer camera photos differ substantially in lighting, resolution, "
                    "and zoom from standardized total-body photography dermoscopy tiles."
                )
        except Exception as e:
            st.error(f"Could not decode image: {e}")


def render_multimodal_inference():
    st.title("🧬 Multimodal Fusion Inference (Image + Metadata)")
    render_warning_banner()

    st.markdown(
        """
        Combines a **$160 \\times 160$ image tensor** with **50 clinical/geometric tabular features**
        through the Multimodal Neural Fusion head ($1,330 \\to 128 \\to 1$).
        """
    )

    col_img, col_meta = st.columns([1, 1])
    with col_img:
        st.subheader("1. Lesion Image")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="mm_img")
        img: Image.Image | None = None
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Target Lesion", width=200)
            except Exception as e:
                st.error(f"Image read error: {e}")

    with col_meta:
        st.subheader("2. Lesion Metadata")
        sex = st.selectbox("Sex", ["male", "female"], key="mm_sex")
        age = st.slider("Age", 10, 90, 50, step=5, key="mm_age")
        site = st.selectbox("Anatomical Site", ["head/neck", "posterior torso", "anterior torso", "lower extremity", "upper extremity"], key="mm_site")
        diam = st.number_input("Diameter (mm)", 0.5, 30.0, 5.0, 0.1, key="mm_diam")
        area = st.number_input("Area (mm²)", 0.5, 200.0, 18.0, 0.5, key="mm_area")
        perim = st.number_input("Perimeter (mm)", 1.0, 150.0, 15.0, 0.5, key="mm_perim")

    weighted_model = st.checkbox("Use Weighted Fusion Model (pos_weight=20.0)", value=True)

    if st.button("Run Multimodal Fusion Inference", use_container_width=True):
        if img is None:
            st.error("Please upload a valid image before running multimodal inference.")
            return

        feature_dict = {
            "sex": sex,
            "age_approx": float(age),
            "anatom_site_general": site,
            "clin_size_long_diam_mm": float(diam),
            "tbp_lv_areaMM2": float(area),
            "tbp_lv_perimeterMM": float(perim),
        }

        with st.spinner("Extracting CNN features + fusing metadata..."):
            res = predict_multimodal(img, feature_dict, weighted=weighted_model)

        st.subheader("Inference Output")
        st.metric(label="Multimodal Model Output Score", value=f"{res['raw_score']:.5f}")
        st.caption(f"Frozen Threshold: `{res['raw_threshold']:.4f}`")
        if res["is_elevated_risk"]:
            st.error("Model Risk Flag: Elevated (Score >= Frozen Threshold)")
        else:
            st.success("Model Risk Flag: Below Threshold")


def render_limitations():
    st.title("⚠️ Limitations, Dataset Context & Ethical Disclaimers")
    render_warning_banner()

    st.markdown(
        """
        ### 1. Extreme Class Imbalance & Statistical Sensitivity
        * In the ISIC 2024 dataset, malignant lesions represent approximately **0.096% of cases** (~1 in 1,040).
        * The held-out test split has **80,210 lesions but only 77 malignant cases**.
        * Small shifts in ranking for just 2–3 patients substantially change the point estimate of Average Precision (AP).
        * Patient cluster bootstrap demonstrates wide, overlapping 95% confidence intervals across all model architectures.

        ### 2. Matched-Cohort Sampling and Calibration
        * To make training feasible on local hardware without memory exhaustion, training utilized a **10,000-lesion matched cohort** with an artificially elevated positive prevalence of **2.390%**.
        * Uncalibrated model scores are systematically inflated above real-world prevalence.
        * Platt scaling successfully recalibrates the expected probability to 0.108% without changing ranking, but must still be interpreted as a research score, not a clinical risk.

        ### 3. Patient Grouping & Leakage Safeguards
        * All train/validation/test partitions were grouped strictly by `patient_id` using `StratifiedGroupKFold(seed=42)`.
        * Zero patient overlap exists across partitions.
        * Post-biopsy pathology, patient IDs, and scanner confidence scores were strictly excluded from feature inputs.

        ### 4. Non-Clinical Disclaimer
        * **SkinRisk Fusion is strictly an educational research prototype.**
        * It is **NOT** a certified Medical Device (SaMD), has not undergone clinical trials, and must never be used for healthcare decision-making.
        """
    )


def main():
    st.sidebar.title("🔬 SkinRisk Fusion")
    st.sidebar.caption("Multimodal Skin Lesion Risk Classification")
    section = st.sidebar.radio(
        "Navigation",
        [
            "Overview & Architecture",
            "Model Comparison & Results",
            "Metadata-Only Inference",
            "Image-Only CNN Inference",
            "Multimodal Fusion Inference",
            "Limitations & Disclaimers",
        ],
    )

    if section == "Overview & Architecture":
        render_overview()
    elif section == "Model Comparison & Results":
        render_comparison()
    elif section == "Metadata-Only Inference":
        render_metadata_inference()
    elif section == "Image-Only CNN Inference":
        render_image_inference()
    elif section == "Multimodal Fusion Inference":
        render_multimodal_inference()
    elif section == "Limitations & Disclaimers":
        render_limitations()


if __name__ == "__main__":
    main()
