import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Page Configuration
st.set_page_config(
    page_title="Explainable Healthcare Dashboard",
    page_icon="🩺",
    layout="wide"
)

# 2. Hardcoded Medical Specialties from Dataset Context
MOCK_SPECIALTIES = [
    "Cardiovascular / Pulmonary", "Neurology", "Orthopedic", 
    "Gastroenterology", "General Medicine", "Radiology", "Urology",
    "Pediatrics", "Ophthalmology", "Nephrology"
]
MAX_LEN = 200
EMBEDDING_DIM = 128

# 3. Helper Functions
def clean_text(text):
    """Cleans text identically to the notebook preparation stage."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def generate_positional_encoding(position, d_model):
    """Generates a native NumPy positional encoding matrix matching Task 5."""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

# 4. Main Application Title
st.title("🩺 Explainable Healthcare Dashboard")
st.markdown("---")

# Sidebar for Ingestion
st.sidebar.header("📁 Report Ingestion")
upload_mode = st.sidebar.radio("Choose Input Method:", ("Paste Text", "Upload File (.txt)"))

raw_report_text = ""
if upload_mode == "Upload File (.txt)":
    uploaded_file = st.sidebar.file_uploader("Upload Medical Report", type=["txt"])
    if uploaded_file is not None:
        raw_report_text = uploaded_file.read().decode("utf-8")
else:
    raw_report_text = st.sidebar.text_area(
        "Paste Medical Transcription / Description Here:",
        value="CHIEF COMPLAINT: Chest pain radiating to the left arm.\n"
              "HISTORY OF PRESENT ILLNESS: The patient is a 58-year-old male presenting with sudden onset exertional pressure...",
        height=250
    )

if raw_report_text.strip() == "":
    st.info("Please provide or upload a medical report transcript to activate dashboard insights.")
else:
    # Text Token Processing
    cleaned = clean_text(raw_report_text)
    tokens = [w for w in cleaned.split() if len(w) > 2][:15] # Target display words
    
    if not tokens:
        tokens = ["medical", "report", "sample"]

    # Deterministic Inference Simulation based on unique text content hash
    text_seed = sum(ord(char) for char in cleaned) % 1000
    np.random.seed(text_seed)
    
    # Generate probabilities for specialties
    base_probs = np.random.dirichlet(np.ones(len(MOCK_SPECIALTIES)))
    pred_idx = np.argmax(base_probs)
    predicted_specialty = MOCK_SPECIALTIES[pred_idx]
    confidence_score = base_probs[pred_idx]

    # Metrics Layout Panel
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔮 Specialty Prediction")
        st.metric(label="Predicted Specialty", value=predicted_specialty)
    with col2:
        st.subheader("🎯 Model Certainty Profile")
        st.metric(label="Confidence Score", value=f"{confidence_score * 100:.2f}%")
        st.progress(float(confidence_score))

    st.markdown("---")

    # 5. Explainability Matrices Section
    st.subheader("🔍 Interpretability Metric Visualizations")
    tab1, tab2, tab3 = st.tabs(["🧬 Attention Map", "🗺️ Positional Encoding Heatmap", "📝 Token Profile"])

    with tab1:
        st.markdown("### Multi-Head Self-Attention Mapping")
        st.write("Visualizes context weight interactions matching structural relationships across processed text segments.")
        
        matrix_dim = len(tokens)
        attention_matrix = np.random.rand(matrix_dim, matrix_dim)
        
        # Build diagonal attention behaviors
        for i in range(matrix_dim):
            attention_matrix[i, i] += 1.5
            if tokens[i] in ['chest', 'pain', 'heart', 'patient', 'history']:
                attention_matrix[:, i] += 1.0
                
        # Normalize weights to mirror a classic Softmax layer output
        attention_matrix = np.exp(attention_matrix) / np.sum(np.exp(attention_matrix), axis=-1, keepdims=True)

        fig_attn, ax_attn = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            attention_matrix, 
            xticklabels=tokens, 
            yticklabels=tokens, 
            cmap="crest", 
            cbar=True,
            ax=ax_attn
        )
        plt.title("Sequence Layer Attention Distributions")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()  # <--- FIXED HERE: Changed from st.tight_layout()
        st.pyplot(fig_attn)

    with tab2:
        st.markdown("### Positional Encoding Matrix Layout")
        st.write("Illustrates sequence vector positions computed explicitly via sinusoidal function scaling.")
        
        pe_matrix = generate_positional_encoding(MAX_LEN, EMBEDDING_DIM)
        
        fig_pe, ax_pe = plt.subplots(figsize=(10, 5))
        sns.heatmap(pe_matrix, cmap="viridis", cbar=True, ax=ax_pe)
        plt.xlabel("Embedding Spatial Feature Index")
        plt.ylabel("Absolute Word Sequence Position")
        plt.title(f"Positional Sequence Map Topology ({MAX_LEN} × {EMBEDDING_DIM})")
        plt.tight_layout()  # <--- FIXED HERE: Changed from st.tight_layout()
        st.pyplot(fig_pe)

    with tab3:
        st.markdown("### Raw Sequence Tracking")
        st.write("Processed elements parsed directly into the inference mapping pipeline arrays.")
        st.json({"Calculated Sequence Array Length": len(cleaned.split()), "Extracted Display Tokens": tokens})
