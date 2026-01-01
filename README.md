# 🌱 AI-Based Carbon Emission Prediction and Behavior Optimization using Reinforcement Learning

**Tech Stack:** Python · Machine Learning · Reinforcement Learning · Streamlit  

An end-to-end AI-driven system that predicts household and individual carbon emissions, explains the driving factors, and recommends personalized lifestyle actions to reduce carbon footprint.

---

## 📌 Project Overview
This project implements a **carbon emission prediction and optimization pipeline** using Machine Learning, Explainable AI, and Reinforcement Learning.  
The system not only estimates carbon emissions accurately but also provides **transparent explanations** and **actionable recommendations** for sustainability.

### Key Goals
- Predict monthly carbon emissions based on lifestyle and energy usage  
- Explain predictions using interpretable AI techniques  
- Recommend low-disruption behavioral changes  
- Promote awareness and sustainable decision-making  

---

## 🧠 System Architecture
```text
User Input
   ↓
Data Preprocessing
   ↓
LightGBM Carbon Prediction
   ↓
SHAP Explainability
   ↓
DQN Reinforcement Learning Agent
   ↓
Personalized Emission Reduction Recommendations
```
---

📁 Project Folder Structure

carbon-emission-optimization/
│
├── Carbon Emission.csv          # Dataset used for training and evaluation
├── train_pipeline.ipynb         # Data preprocessing, model training & evaluation
├── app.py                       # Streamlit web application
├── dqn_carbon_agent_final.zip   # Trained DQN reinforcement learning agent
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation

---

📊 Dataset Description

Size: ~100,000 anonymized records

Features include:

Demographics (age group, household size)

Energy usage (electricity, heating source, renewables)

Mobility (vehicle type, travel distance, flights)

Lifestyle & waste (diet type, recycling, waste generation)

Target Variable:

Monthly carbon emission (kg CO₂e)

---

🧪 Methodology

1️⃣ Data Preprocessing

Missing value imputation (mean/mode)

One-hot and ordinal encoding

Outlier handling

Train/validation/test split (80/10/10)

2️⃣ Carbon Emission Prediction

Models evaluated:

Linear Regression

Random Forest

Gradient Boosting

Best Model: LightGBM

Performance:

R² ≈ 0.97

Low MAE and RMSE

3️⃣ Explainable AI (SHAP)

Global and local feature importance

Identifies major emission drivers:

Vehicle usage

Energy source

Diet type

Electricity consumption

4️⃣ Reinforcement Learning Optimization

Algorithm: Deep Q-Network (DQN)

Actions include:

Reduce private vehicle usage

Switch to public transport

Adopt renewable energy

Reduce meat consumption

Improve recycling habits

Outcome:

~12–22% emission reduction in simulation

---

🌐 Streamlit Web Application

The Streamlit dashboard allows users to:

Enter lifestyle and energy usage data

View predicted carbon emissions

Understand influencing factors via SHAP plots

Receive personalized emission-reduction recommendations

---

🛠️ Technologies Used

Python

LightGBM

Scikit-learn

SHAP

TensorFlow / Keras

Reinforcement Learning (DQN)

Streamlit

Pandas, NumPy, Matplotlib

---

📈 Key Results

High-accuracy carbon emission prediction

Transparent and interpretable model decisions

Intelligent, adaptive recommendations using RL

Practical real-world sustainability application

---

🚀 Future Enhancements

Integration with IoT and smart meters

Federated learning for privacy-preserving training

Multi-agent reinforcement learning for community-level optimization

Real-world deployment and feedback-based learning

---

👨‍💻 Author

Ushodaya Dasari
M.Tech – Artificial Intelligence & Machine Learning
(In Collaboration with LTIMindtree)
Vellore Institute of Technology, Vellore
