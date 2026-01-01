# AI-Based Carbon Emission Prediction and Behavior Optimization using Reinforcement Learning

## Project Overview
This project presents an end-to-end AI-based system to predict individual and household carbon emissions and recommend personalized lifestyle changes to reduce carbon footprint. The system integrates supervised machine learning, explainable AI, and reinforcement learning to provide accurate, transparent, and actionable sustainability insights.

A Streamlit-based web application is used to interact with users and display predictions, explanations, and optimization suggestions.

---

## Objectives
- Predict monthly carbon emissions using lifestyle and energy usage data  
- Explain predictions using interpretable AI techniques  
- Recommend personalized actions to reduce emissions  
- Demonstrate real-world application of ML + XAI + RL  

---

## System Workflow
User Input → Data Preprocessing → LightGBM Prediction → SHAP Explainability → DQN Optimization → Recommendations

---

## Repository Structure
├── Carbon Emission.csv # Dataset
├── train_pipeline.ipynb # Data preprocessing and model training
├── app.py # Streamlit application
├── dqn_carbon_agent_final.zip # Trained DQN agent
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## Dataset Description
- Approximately 100,000 anonymized records  
- Features include:
  - Demographics
  - Energy consumption
  - Transportation habits
  - Diet and waste management
- Target variable:
  - Monthly carbon emission (kg CO₂e)

---

## Methodology

### Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling and outlier handling
- Train/validation/test split

### Carbon Emission Prediction
- Models tested: Linear Regression, Random Forest, Gradient Boosting
- Best model: LightGBM
- Achieved high prediction accuracy (R² ≈ 0.97)

### Explainable AI (SHAP)
- Identifies key contributors to carbon emissions
- Provides both global and individual-level explanations

### Reinforcement Learning Optimization
- Algorithm used: Deep Q-Network (DQN)
- Suggests actions such as:
  - Reducing private vehicle usage
  - Switching to public transport
  - Adopting renewable energy
  - Improving recycling habits
- Achieved 12–22% emission reduction in simulation

---

## Streamlit Application
The Streamlit dashboard allows users to:
- Enter lifestyle details
- View predicted carbon emissions
- Understand influencing factors through SHAP plots
- Receive personalized recommendations

---

## Installation and Usage

### Clone the Repository
```bash
git clone https://github.com/Ushodaya07/AI-Based-Carbon-Emission-Prediction-and-Behavior-Optimization-using-Reinforcement-Learning.git
cd AI-Based-Carbon-Emission-Prediction-and-Behavior-Optimization-using-Reinforcement-Learning
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
