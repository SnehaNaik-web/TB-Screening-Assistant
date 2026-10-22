import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import logging

# Clean logging for runtime errors without exposing tracebacks in the interface
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tb_screening")

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="TB Screening Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MODEL SETUP & PREPROCESSING ----------------
MODEL_PATH = Path(__file__).parent / "densenet121_combined_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads the trained DenseNet-121 binary classification model checkpoint.
    Architecture, weights, and inference logic are strictly preserved.
    """
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# ImageNet normalization pipeline (identical to training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- STATE MANAGEMENT ----------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None

if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

def reset_screening():
    """Resets file uploader and screening result to clean state."""
    st.session_state.uploader_key += 1
    st.session_state.screening_result = None
    st.session_state.last_processed_file = None

# ---------------- CLEAN, PROFESSIONAL HEALTHCARE STYLING ----------------
st.markdown("""
<style>
/* Universal Background & Typography (Pure White, Dark Charcoal) */
html, body, .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #ffffff !important;
    color: #262626 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Force Clean White Sidebar (Eliminates Dark Sidebar) */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarNav"] {
    background-color: #ffffff !important;
    color: #262626 !important;
    border-right: 1px solid #e5e7eb !important;
}

/* Streamlit Header & Chrome Cleanup */
header[data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 1px solid #e5e7eb !important;
}
.stAppDeployButton, #MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

/* Layout Spacing */
.main .block-container {
    max-width: 1080px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* App Header Section */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.header-titles h1 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #171717;
    margin: 0;
    letter-spacing: -0.01em;
}
.header-titles p {
    font-size: 0.84rem;
    color: #525252;
    margin: 0.2rem 0 0 0;
}
.status-tag {
    font-size: 0.76rem;
    font-weight: 500;
    color: #525252;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 0.25rem 0.6rem;
    border-radius: 3px;
    white-space: nowrap;
}

/* Section Headings */
.section-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #171717;
    margin: 0 0 0.2rem 0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.section-desc {
    font-size: 0.82rem;
    color: #525252;
    margin: 0 0 0.85rem 0;
    line-height: 1.4;
}

/* Clean Image Preview Container */
.image-preview-frame {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 0.5rem;
    margin-bottom: 0.65rem;
    display: flex;
    justify-content: center;
}
.meta-line {
    font-size: 0.78rem;
    color: #525252;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.meta-pill {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
}

/* Simple, Plain Notice (Replaces large colored banners) */
.plain-notice {
    font-size: 0.78rem;
    color: #525252;
    border-left: 3px solid #0f766e;
    padding: 0.4rem 0.65rem;
    margin-bottom: 1rem;
    background: #ffffff;
    line-height: 1.4;
}

/* Clean Results Section (No loud colored alert banners) */
.result-box {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    background: #ffffff;
}
.result-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.result-kicker {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #525252;
}
.result-heading-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: #171717;
    margin: 0 0 0.35rem 0;
}
.result-explanation-text {
    font-size: 0.82rem;
    color: #525252;
    margin: 0 0 1rem 0;
    line-height: 1.45;
}
.result-divider {
    border-top: 1px solid #e5e7eb;
    margin: 0.85rem 0;
}
.confidence-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}
.confidence-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #262626;
}
.confidence-figure {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f766e;
    font-variant-numeric: tabular-nums;
}
.confidence-note {
    font-size: 0.76rem;
    color: #6b7280;
    margin-top: 0.4rem;
    line-height: 1.4;
}

/* Probability Table */
.index-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    margin-top: 0.65rem;
}
.index-table td {
    padding: 0.3rem 0;
    border-bottom: 1px solid #f3f4f6;
    color: #525252;
}
.index-table .val-col {
    text-align: right;
    font-weight: 500;
    color: #171717;
}

/* Awaiting Placeholder */
.idle-box {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 2.25rem 1.25rem;
    text-align: center;
    background: #ffffff;
}
.idle-box-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #262626;
    margin-bottom: 0.3rem;
}
.idle-box-desc {
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.45;
    max-width: 320px;
    margin: 0 auto;
}

/* Buttons (Muted Teal Primary, Clean White Secondary) */
div.stButton > button {
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 500;
    border-radius: 4px;
    padding: 0.45rem 1rem;
    border: 1px solid #d1d5db;
}
div.stButton > button[kind="primary"] {
    background-color: #0f766e !important;
    color: #ffffff !important;
    border-color: #0f766e !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #115e59 !important;
    border-color: #115e59 !important;
}
div.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #262626 !important;
    border-color: #d1d5db !important;
}
div.stButton > button[kind="secondary"]:hover {
    background-color: #f9fafb !important;
    color: #171717 !important;
    border-color: #9ca3af !important;
}

/* Clean Medical Disclaimer Section */
.disclaimer-container {
    border-top: 1px solid #e5e7eb;
    padding-top: 1rem;
    margin-top: 2rem;
}
.disclaimer-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #525252;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.disclaimer-body {
    font-size: 0.77rem;
    color: #525252;
    line-height: 1.45;
    margin: 0;
}

/* Sidebar Sections (Clean spaced text, no cards or dark backgrounds) */
.sidebar-group {
    margin-bottom: 1.25rem;
}
.sidebar-group-title {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #171717;
    margin-bottom: 0.65rem;
}
.spec-entry {
    margin-bottom: 0.65rem;
}
.spec-entry-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.1rem;
}
.spec-entry-value {
    font-size: 0.82rem;
    font-weight: 500;
    color: #262626;
}
.sidebar-separator {
    border-top: 1px solid #e5e7eb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APPLICATION HEADER ----------------
st.markdown("""
<div class="top-header">
    <div class="header-titles">
        <h1>TB Screening Assistant</h1>
        <p>Chest radiograph pattern analysis prototype for pulmonary tuberculosis screening</p>
    </div>
    <div class="status-tag">
        Local Execution • Standalone Offline
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR SPECIFICATIONS ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-group">
        <div class="sidebar-group-title">System Specifications</div>
        <div class="spec-entry">
            <div class="spec-entry-label">Model Architecture</div>
            <div class="spec-entry-value">DenseNet-121 (Binary Classifier)</div>
        </div>
        <div class="spec-entry">
            <div class="spec-entry-label">Screening Task</div>
            <div class="spec-entry-value">Normal vs. TB-Consistent Pattern</div>
        </div>
        <div class="spec-entry">
            <div class="spec-entry-label">Inference Runtime</div>
            <div class="spec-entry-value">PyTorch (Local """ + str(DEVICE).upper() + """)</div>
        </div>
    </div>
    <div class="sidebar-separator"></div>
    <div class="sidebar-group">
        <div class="sidebar-group-title">Data Privacy & Execution</div>
        <div class="spec-entry">
            <div class="spec-entry-label">Network Mode</div>
            <div class="spec-entry-value">100% Offline / Standalone</div>
        </div>
        <div class="spec-entry">
            <div class="spec-entry-label">Telemetry & Storage</div>
            <div class="spec-entry-value">In-memory computation only. No patient images or logs are stored or transmitted.</div>
        </div>
    </div>
    <div class="sidebar-separator"></div>
    <div class="sidebar-group">
        <div class="sidebar-group-title">Input Criteria & Scope</div>
        <div class="spec-entry">
            <div class="spec-entry-label">Required Modality</div>
            <div class="spec-entry-value">Frontal (PA) Chest Radiographs</div>
        </div>
        <div class="spec-entry">
            <div class="spec-entry-label">Accepted Formats</div>
            <div class="spec-entry-value">PNG, JPG, JPEG</div>
        </div>
        <div class="spec-entry">
            <div class="spec-entry-label">Operational Boundary</div>
            <div class="spec-entry-value">Does not diagnose tuberculosis or reliably identify non-chest images.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN WORKFLOW LAYOUT ----------------
col_left, col_right = st.columns([1.05, 0.95], gap="large")

# ==================== LEFT COLUMN: INPUT & PREVIEW ====================
with col_left:
    st.markdown("""
    <div class="section-title">Chest Radiograph Input</div>
    <div class="section-desc">Upload a standard frontal (PA) chest X-ray image for screening analysis.</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload radiograph image",
        type=["png", "jpg", "jpeg"],
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.session_state.last_processed_file != uploaded_file.name:
            st.session_state.screening_result = None

        try:
            image = Image.open(uploaded_file).convert("RGB")

            # Image Preview Container
            st.markdown('<div class="image-preview-frame">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Metadata Display
            file_size_kb = len(uploaded_file.getvalue()) / 1024.0
            file_ext = uploaded_file.type.split('/')[-1].upper() if uploaded_file.type else "IMAGE"
            st.markdown(f"""
            <div class="meta-line">
                <span class="meta-pill">Dimensions: {image.width} × {image.height} px</span>
                <span class="meta-pill">Size: {file_size_kb:.1f} KB</span>
                <span class="meta-pill">Format: {file_ext}</span>
            </div>
            """, unsafe_allow_html=True)

            # Plain input limitation notice (No large colored banner)
            st.markdown("""
            <div class="plain-notice">
                Notice: This model is trained specifically on chest radiographs.
                It does not validate anatomical positioning and cannot reliably identify non-chest images.
            </div>
            """, unsafe_allow_html=True)

            # Action Controls
            btn_col1, btn_col2 = st.columns([1.4, 1.0])
            with btn_col1:
                run_btn = st.button("Run Screening", type="primary", use_container_width=True)
            with btn_col2:
                remove_btn = st.button("Remove Image", type="secondary", use_container_width=True, on_click=reset_screening)

            if run_btn:
                if not MODEL_PATH.exists():
                    st.error("Model checkpoint not found. Ensure 'densenet121_combined_best.pth' is located in the application directory.")
                else:
                    with st.spinner("Analyzing radiograph locally..."):
                        try:
                            model = load_model()
                            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

                            with torch.inference_mode():
                                outputs = model(input_tensor)
                                probabilities = torch.softmax(outputs, dim=1)[0]
                                predicted_class = int(torch.argmax(probabilities).item())
                                confidence = float(probabilities[predicted_class].item())
                                probs_list = [float(p.item()) for p in probabilities]

                            st.session_state.screening_result = {
                                "class_idx": predicted_class,
                                "confidence": confidence,
                                "probs": probs_list,
                                "filename": uploaded_file.name
                            }
                            st.session_state.last_processed_file = uploaded_file.name
                            st.rerun()

                        except Exception as e:
                            logger.error(f"Inference error: {str(e)}", exc_info=True)
                            st.error("An error occurred during image processing. Please ensure the image is a valid chest radiograph and retry.")

        except (UnidentifiedImageError, OSError):
            st.error("Unable to decode the uploaded file. Please verify that it is an undamaged PNG or JPEG image.")
        except Exception as e:
            logger.error(f"File reading error: {str(e)}", exc_info=True)
            st.error("Failed to load image. Please select a valid chest radiograph.")
    else:
        st.caption("Accepted formats: PNG, JPG, JPEG • Frontal (PA) chest radiographs only")

# ==================== RIGHT COLUMN: RESULTS & ASSESSMENT ====================
with col_right:
    st.markdown("""
    <div class="section-title">Screening Assessment</div>
    <div class="section-desc">Automated feature evaluation generated by the local screening model.</div>
    """, unsafe_allow_html=True)

    result = st.session_state.screening_result

    if result is not None:
        is_normal = (result["class_idx"] == 0)
        confidence_pct = result["confidence"] * 100.0
        normal_pct = result["probs"][0] * 100.0
        abnormal_pct = result["probs"][1] * 100.0

        if is_normal:
            classification_title = "Normal-Pattern Classification"
            classification_desc = "The model did not identify radiographic patterns characteristic of pulmonary tuberculosis in this image."
            classification_kicker = "Screening Result: Negative"
        else:
            classification_title = "Abnormal-Pattern Classification"
            classification_desc = "The model identified radiographic patterns consistent with pulmonary tuberculosis or chest abnormality. Clinical review recommended."
            classification_kicker = "Screening Result: Review Advised"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-header-row">
                <span class="result-kicker">{classification_kicker}</span>
            </div>
            <div class="result-heading-text">{classification_title}</div>
            <div class="result-explanation-text">{classification_desc}</div>
            <div class="result-divider"></div>
            <div class="confidence-meta-row">
                <span class="confidence-title">Model Output Confidence</span>
                <span class="confidence-figure">{confidence_pct:.1f}%</span>
            </div>
            <div class="confidence-note">
                Confidence represents the mathematical certainty of the model's output layer for the classified pattern.
                It is not the clinical probability that a patient is healthy or infected with tuberculosis.
            </div>
            <table class="index-table">
                <tr>
                    <td>Normal Pattern Index</td>
                    <td class="val-col">{normal_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>TB-Consistent Abnormality Index</td>
                    <td class="val-col">{abnormal_pct:.1f}%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.button("Screen Another Image", type="secondary", use_container_width=True, on_click=reset_screening)

    else:
        st.markdown("""
        <div class="idle-box">
            <div class="idle-box-title">Awaiting Radiograph Analysis</div>
            <div class="idle-box-desc">
                Upload a chest X-ray on the left and select <strong>Run Screening</strong> to evaluate patterns with the DenseNet-121 classifier.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- MEDICAL & RESEARCH DISCLAIMER FOOTER ----------------
st.markdown("""
<div class="disclaimer-container">
    <div class="disclaimer-header">Research Prototype Notice & Clinical Disclaimer</div>
    <p class="disclaimer-body">
        This application is an artificial intelligence screening prototype developed for research, benchmarking, and investigational evaluation only.
        It is not an FDA-cleared, CE-marked, or clinically certified medical diagnostic device. It does not provide medical diagnoses, treatment plans, or clinical directives.
        It does not reliably identify non-chest images. Findings must always be interpreted in conjunction with complete clinical history, bacteriological examination (e.g., sputum smear/culture, GeneXpert), and direct evaluation by a qualified medical professional.
    </p>
</div>
""", unsafe_allow_html=True)
