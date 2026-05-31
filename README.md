# 🏭 Quality Control Vision System (Multi-Task CNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikamsudarshan/quality_control_vision_system/blob/main/Model_Training_Pipeline.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

An enterprise-grade, real-time visual inspection system built for manufacturing assembly lines. This project utilizes a **Multi-Task Learning Convolutional Neural Network** to simultaneously classify industrial surface defects and perform regression to determine defect severity, triggering automated rejection logic via an operator dashboard.

## 🚀 Project Overview

Modern mechatronics and manufacturing pipelines require intelligent quality control at the edge. This system processes camera feeds of metal components and makes split-second decisions on part viability. 

### Key Engineering Features
* **Multi-Task Learning:** Instead of running two separate models, the architecture utilizes a shared `EfficientNetB0` backbone that splits into two parallel heads:
  * A `Softmax` classification head (identifying defect type).
  * A `Sigmoid` regression head (calculating severity from 0.0 to 1.0).
* **Dynamic Feature Engineering:** Bypassed the need for manual bounding-box annotations by engineering a custom "Severity Score" proxy calculated via pixel standard deviation during the data pipeline mapping phase.
* **Thread-Safe Architecture:** Engineered for stable deployment on Linux systems by strictly isolating the TensorFlow C++ XLA compiler from the Streamlit asynchronous web server via "Lazy Loading," preventing memory leaks and segmentation faults.
* **Developer Telemetry:** Includes an in-memory JSON telemetry report for operators to view full probability distributions of edge-case parts.

## 📸 Dashboard Preview


## 🧠 Model Architecture & Training

You can view, run, and modify the entire training pipeline directly in your browser using the Google Colab link at the top of this page.

The model is trained on the **NEU Surface Defect Database** (1,800 images of hot-rolled steel strips).
1. **Inputs:** `224x224x3` RGB Tensors.
2. **Backbone:** Frozen Google EfficientNet (Optimized for low-latency edge computing).
3. **Loss Functions:** Sparse Categorical Crossentropy (Classification) + Mean Squared Error (Regression).

## 💻 How to Run the Dashboard Locally

The web application is strictly configured to run in CPU-only mode to guarantee thread stability on all operating systems without requiring proprietary GPU drivers.

1. Clone this repository:
```bash
git clone https://github.com/nikamsudarshan/quality_control_vision_system.git
cd quality-control-vision-system
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the Operator Dashboard:
```bash
streamlit run app.py
```
