# HPC-Driven Predictive Healthcare Analytics using AI & LLMs

## Project Overview
This project presents an HPC-enabled predictive healthcare analytics system for diabetes risk assessment. 
It integrates hybrid ensemble machine learning, GPU acceleration, MPI-based parallelism, and Large Language Models (LLMs) to deliver accurate, scalable, and interpretable clinical predictions.

The system is designed as a production-ready healthcare analytics platform capable of high-throughput patient screening.

---

## Key Features
- Hybrid ensemble model (Logistic Regression, XGBoost, Deep Neural Network)
- Dual-GPU acceleration with CUDA
- MPI-based parallel training using MPI4Py
- 2.42× speedup over CPU baseline
- LLM-generated clinical explanations
- Interactive Streamlit dashboard for real-time analytics

---

## Dataset
- Synthetic diabetes dataset (HIPAA-compliant)
- 100,000 patient records
- 24 clinical, demographic, and lifestyle features
- Binary classification: Diabetic / Non-Diabetic

---

## Technology Stack
- Python 3.10+
- MPI4Py, OpenMPI
- CUDA, cuDNN
- TensorFlow (GPU)
- XGBoost (GPU)
- scikit-learn
- Hugging Face LLaMA (LLM)
- Streamlit
- Pandas, NumPy, Matplotlib

---

## System Architecture
- MPI Rank 0: DNN (GPU-0) + Logistic Regression (CPU)
- MPI Rank 1: XGBoost (GPU-1)
- Weighted soft-voting ensemble
- LLM explainability layer for clinical interpretation

---

## Performance Highlights
- Hybrid Model Accuracy: **91.90%**
- Dual-GPU Parallel Speedup: **2.42×**
- Precision (Diabetic): **100%**
- Recall (Diabetic): **87%**

---

## Folder Structure

<pre>
hpc-predictive-healthcare-analytics/
├── README.md
│
├── data/
│   └── diabetes_dataset.csv
│
├── code/
│   └── final-project.ipynb
│
├── models/
│   └── best_diabetes_model_HPC.pkl
│
├── reports/
│   └── Final_Project_Report.pdf
│
└── dashboard/
    └── streamlit_app.py
</pre>

---

## Dashboard Features
- Single-patient prediction
- Batch CSV upload
- Session-level analytics
- HPC performance metrics
- LLM-generated clinical summaries

---

## Author
Vaibhav/vee-07

---

## Note
This project was developed as part of the Advanced Certificate Course (ACC) in High Performance Computing (HPC) at C-DAC ACTS, Pune.
