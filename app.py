"""
🏭 Quality Control Vision System - Operator Dashboard
A Streamlit web interface for real-time inference of industrial defects.
Implements 'Lazy Loading' to ensure thread-safety on Linux/Unix systems.
"""

import os
# --- ARCHITECTURAL SAFEGUARDS ---
# Block CUDA at the OS level to prevent C++ XLA compiler segmentation faults
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress C++ hardware acceleration warnings in the terminal
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# ==========================================
# System Configuration
# ==========================================
st.set_page_config(page_title="Vision QC System", page_icon="🏭", layout="wide")

CLASS_NAMES = ['Crazing', 'Inclusions', 'Patches', 'Pitted', 'Rolled-in Scale', 'Scratches']
IMG_SIZE = 224

# ==========================================
# UI Layout
# ==========================================
st.title("🏭 Quality Control Vision System")
st.markdown("Automated surface defect detection and severity grading using a Multi-Output CNN.")

tab1, tab2 = st.tabs(["📷 Inspection Line", "📊 Analytics Dashboard"])

with tab1:
    st.header("Real-Time Part Inspection")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Camera Feed")
        uploaded_file = st.file_uploader("Upload Part Image", type=["jpg", "png", "bmp"])

        st.markdown("### Operator Settings")
        reject_threshold = st.slider("Rejection Severity Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    with col2:
        st.subheader("System Output")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Current Part on Conveyor", use_column_width=True)

            # Data Preprocessing
            img_array = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            img_batch = np.expand_dims(img_resized, axis=0)

            # ==========================================
            # LAZY-LOADED AI ENGINE
            # ==========================================
            # By importing TensorFlow strictly inside this execution block, we isolate
            # its C++ memory allocations from Streamlit's asynchronous web server threads.
            with st.spinner("Powering up AI Engine and scanning..."):
                import tensorflow as tf

                # Constrain CPU threading locally to prevent RAM choke
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.set_visible_devices([], 'GPU')

                # Load weights and run inference
                model = tf.keras.models.load_model('vision_qc_model.keras')
                predictions = model.predict(img_batch)
            # ==========================================

            # Parse Multi-Output Predictions
            class_probs = predictions[0][0]
            severity_score = float(predictions[1][0][0])
            top_class_idx = np.argmax(class_probs)
            predicted_defect = CLASS_NAMES[top_class_idx]

            # Operator Logic Gate
            st.markdown("---")
            if severity_score >= reject_threshold:
                st.error(f"🚨 **REJECT TRIGGERED**")
                st.write(f"**Defect Type:** {predicted_defect}")
                st.write(f"**Severity Score:** {severity_score:.3f} (Exceeds {reject_threshold} threshold)")
            else:
                st.success(f"✅ **PASS**")
                st.write(f"**Detected Surface:** {predicted_defect} (Minor/Acceptable)")
                st.write(f"**Severity Score:** {severity_score:.3f}")

            # Developer Memory Log
            with st.expander("🛠️ View Raw Developer Report"):
                report = {
                    "predicted_class": predicted_defect,
                    "regression_severity": round(severity_score, 4),
                    "probability_distribution": {
                        CLASS_NAMES[i]: f"{float(class_probs[i])*100:.2f}%" for i in range(len(CLASS_NAMES))
                    }
                }
                st.json(report)

with tab2:
    st.header("Shift Analytics & Tracking")

    # Simulated database payload for dashboard visualization
    mock_data = pd.DataFrame({
        "Machine": ["SCARA-1", "SCARA-1", "Conveyor-A", "Conveyor-A", "SCARA-1"],
        "Defect Type": ["Scratches", "Pitted", "Scratches", "Inclusions", "Crazing"],
        "Severity": [0.82, 0.41, 0.91, 0.22, 0.76],
        "Action": ["REJECT", "PASS", "REJECT", "PASS", "REJECT"]
    })

    st.dataframe(mock_data, use_container_width=True)
    st.subheader("Defect Frequency by Type")
    st.bar_chart(mock_data['Defect Type'].value_counts())
