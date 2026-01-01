AI-Based Carbon Emission Prediction and Behavior Optimization using Reinforcement Learning
📌 Project Overview

This project presents an end-to-end AI-driven framework to predict household/individual carbon emissions and recommend personalized lifestyle changes to reduce emissions.

The system combines:

Supervised Machine Learning for accurate carbon footprint prediction

Explainable AI (XAI) for transparent and interpretable predictions

Reinforcement Learning (RL) for adaptive behavior optimization

A Streamlit web application is used to make the system interactive and user-friendly.

🎯 Objectives

Predict monthly carbon emissions based on lifestyle, energy usage, and mobility data

Explain prediction results using feature-level interpretability

Recommend low-disruption, personalized actions to reduce emissions

Demonstrate how ML + XAI + RL can work together for sustainability

🧠 System Architecture

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

##📂 Repository Structure

├── Carbon Emission.csv          # Dataset used for training and evaluation
├── train_pipeline.ipynb         # Data preprocessing, model training & evaluation
├── app.py                       # Streamlit web application
├── dqn_carbon_agent_final.zip   # Trained DQN reinforcement learning agent
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
📊 Dataset Description

Size: ~100,000 anonymized records

Features Include:

Demographics (age group, household size)

Energy usage (electricity, heating source, renewables)

Mobility (vehicle type, travel distance, flights)

Lifestyle & waste (diet type, recycling, waste generation)

Target Variable:

Monthly carbon emission (kg CO₂e)

🧪 Methodology
1️⃣ Data Preprocessing

Missing value imputation (mean/mode)

One-hot & ordinal encoding

Outlier handling

Train/validation/test split (80/10/10)

2️⃣ Carbon Emission Prediction

Models evaluated: Linear Regression, Random Forest, Gradient Boosting

Best Model: LightGBM

Performance:

R² ≈ 0.97

Low MAE and RMSE

3️⃣ Explainable AI (SHAP)

Global and local feature importance

Identifies key emission drivers such as:

Vehicle usage

Energy source

Diet type

Electricity consumption

4️⃣ Reinforcement Learning Optimization

Algorithm: Deep Q-Network (DQN)

Actions Include:

Reduce private vehicle usage

Switch to public transport

Adopt renewable energy

Reduce meat consumption

Improve recycling habits

Outcome:

~12–22% emission reduction in simulation

🌐 Streamlit Web Application

The Streamlit app allows users to:

Enter lifestyle and energy data

View predicted carbon emissions

Understand influencing factors via SHAP plots

Receive personalized emission-reduction suggestions

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Ushodaya07/AI-Based-Carbon-Emission-Prediction-and-Behavior-Optimization-using-Reinforcement-Learning.git
cd AI-Based-Carbon-Emission-Prediction-and-Behavior-Optimization-using-Reinforcement-Learning

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

🛠️ Technologies Used

Python

LightGBM

Scikit-learn

SHAP

TensorFlow / Keras

Reinforcement Learning (DQN)

Streamlit

Pandas, NumPy, Matplotlib

📈 Key Results

High-accuracy carbon emission prediction

Transparent and interpretable model decisions

Intelligent, adaptive recommendations via RL

Practical, real-world sustainability application

🚀 Future Enhancements

Integration with IoT and smart meters

Federated learning for privacy-preserving training

Multi-agent RL for community-level optimization

Real-world deployment and user feedback loop

👤 Author

Ushodaya Dasari
M.Tech – Artificial Intelligence & Machine Learning (in Collaboration with LTIMindtree)
Vellore Institute of Technology, Vellore

