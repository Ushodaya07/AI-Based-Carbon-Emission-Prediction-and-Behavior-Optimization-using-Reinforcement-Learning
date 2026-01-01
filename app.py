import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Optional RL import
try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

# Paths
MODEL_PATH = "models/carbon_rf_pipeline_final.joblib"
AGENT_PATH = "models/dqn_carbon_agent_final.zip"

# RL Actions
ACTIONS = [
    "no_change",
    "reduce_car_frequency",
    "switch_to_public_transport",
    "reduce_meat_consumption",
    "switch_to_renewable_energy",
    "increase_recycling",
]

# -------------------------------
# Feature Engineering
# -------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    rename_map = {
        'Body Type':'Body_Type',
        'How Often Shower':'How_Often_Shower',
        'Heating Energy Source':'Heating_Energy_Source',
        'Vehicle Type':'Vehicle_Type',
        'Social Activity':'Social_Activity',
        'Monthly Grocery Bill':'Monthly_Grocery_Bill',
        'Frequency of Traveling by Air':'Frequency_of_Traveling_by_Air',
        'Vehicle Monthly Distance Km':'Vehicle_Monthly_Distance_Km',
        'Waste Bag Size':'Waste_Bag_Size',
        'Waste Bag Weekly Count':'Waste_Bag_Weekly_Count',
        'How Long TV PC Daily Hour':'How_Long_TV_PC_Daily_Hour',
        'How Many New Clothes Monthly':'How_Many_New_Clothes_Monthly',
        'How Long Internet Daily Hour':'How_Long_Internet_Daily_Hour',
        'Energy efficiency':'Energy_efficiency'
    }
    df = df.rename(columns=rename_map)

    defaults = dict(
        Body_Type="average",
        Sex="other",
        Diet="omnivore",
        How_Often_Shower="daily",
        Heating_Energy_Source="gas",
        Transport="private",
        Vehicle_Type="petrol",
        Social_Activity="sometimes",
        Monthly_Grocery_Bill=200,
        Frequency_of_Traveling_by_Air="rarely",
        Vehicle_Monthly_Distance_Km=0,
        Waste_Bag_Size="medium",
        Waste_Bag_Weekly_Count=2,
        How_Long_TV_PC_Daily_Hour=2,
        How_Many_New_Clothes_Monthly=2,
        How_Long_Internet_Daily_Hour=3,
        Energy_efficiency="medium",
        Recycling="",
        Cooking_With="gas",
    )
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna(v)

    lower_cols = [
        'Diet','How_Often_Shower','Heating_Energy_Source','Transport',
        'Vehicle_Type','Social_Activity','Frequency_of_Traveling_by_Air',
        'Waste_Bag_Size','Energy_efficiency','Recycling','Cooking_With',
        'Body_Type','Sex'
    ]
    for c in lower_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    def recycling_count(s):
        if pd.isna(s) or str(s).strip()=="":
            return 0
        return len([p for p in str(s).split("|") if p.strip()])
    df["Recycling_Count"] = df["Recycling"].apply(recycling_count)

    keep = [
        'Body_Type','Sex','Diet','How_Often_Shower','Heating_Energy_Source','Transport','Vehicle_Type',
        'Social_Activity','Monthly_Grocery_Bill','Frequency_of_Traveling_by_Air','Vehicle_Monthly_Distance_Km',
        'Waste_Bag_Size','Waste_Bag_Weekly_Count','How_Long_TV_PC_Daily_Hour','How_Many_New_Clothes_Monthly',
        'How_Long_Internet_Daily_Hour','Energy_efficiency','Recycling_Count','Cooking_With'
    ]
    return df[keep]

# -------------------------------
# Load Models
# -------------------------------
pipeline = joblib.load(MODEL_PATH)
agent = DQN.load(AGENT_PATH) if SB3_AVAILABLE and os.path.exists(AGENT_PATH) else None

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌍 Carbon Footprint Reduction Assistant", layout="wide")
st.title("🌍 Carbon Footprint Reduction Assistant")
st.write("Predict your carbon footprint and get actionable recommendations to reduce it.")

# Sidebar Inputs
st.sidebar.header("📝 Lifestyle Inputs")
diet = st.sidebar.selectbox("Diet", ["omnivore","vegetarian","vegan","plant-based"])
transport = st.sidebar.selectbox("Transport", ["private","public","walk/bicycle"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["petrol","diesel","electric","none"])
distance = st.sidebar.slider("Vehicle Monthly Distance (Km)", 0, 10000, 500, 100)
energy = st.sidebar.selectbox("Heating Energy Source", ["coal","gas","renewable","electric","solar","wood"])
efficiency = st.sidebar.selectbox("Energy Efficiency", ["no","sometimes","yes"])
recycling = st.sidebar.multiselect("Recycling", ["paper","plastic","glass","metal"])
shower = st.sidebar.selectbox("How Often Shower", ["daily","every 2 days","weekly","twice a day","more frequently","less frequently"])
clothes = st.sidebar.slider("How Many New Clothes Monthly", 0, 50, 2, 1)

row = {
    "Diet": diet,
    "Transport": transport,
    "Vehicle Type": vehicle_type,
    "Vehicle Monthly Distance Km": distance,
    "Heating Energy Source": energy,
    "Energy efficiency": efficiency,
    "Recycling": "|".join(recycling),
    "How Often Shower": shower,
    "How Many New Clothes Monthly": clothes,
    "Body Type": "average",
    "Sex": "other",
    "Social Activity": "sometimes",
    "Monthly Grocery Bill": 200,
    "Frequency of Traveling by Air": "rarely",
    "Waste Bag Size": "medium",
    "Waste Bag Weekly Count": 2,
    "How Long TV PC Daily Hour": 2,
    "How Long Internet Daily Hour": 3,
    "Cooking_With": "gas",
}
df_input = pd.DataFrame([row])

# -------------------------------
# Baseline Prediction
# -------------------------------
feat = build_features(df_input)
baseline = pipeline.predict(feat)[0]

st.subheader("📊 Baseline Emission Prediction")
st.write(f"Estimated carbon emission: **{baseline:.2f} units**")

# -------------------------------
# Column-by-Column Review
# -------------------------------
st.subheader("🔍 Lifestyle Review & Suggestions")
reviews = []

if row["Transport"] == "private":
    reviews.append("⚠️ Consider switching from private transport to public/bicycle/walk.")
else:
    reviews.append("✅ Transport choice is eco-friendly.")

if row["Diet"] in ["omnivore","meat"]:
    reviews.append("⚠️ High-meat diet detected. Reduce meat consumption for ~20-30% emission cut.")
else:
    reviews.append("✅ Diet is low-emission.")

if row["How Many New Clothes Monthly"] > 5:
    reviews.append(f"⚠️ Clothes purchases = {row['How Many New Clothes Monthly']}/month. Try reducing to ≤ 3.")
else:
    reviews.append("✅ Clothes purchases are reasonable.")

if row["Heating Energy Source"] in ["coal","gas","wood"]:
    reviews.append(f"⚠️ Using {row['Heating Energy Source']} for heating. Switch to renewable/solar.")
else:
    reviews.append("✅ Heating source is sustainable.")

if row["Energy efficiency"] in ["no","low"]:
    reviews.append("⚠️ Home not energy efficient. Upgrade appliances/insulation.")
else:
    reviews.append("✅ Energy efficiency is good.")

if len(recycling) == 0:
    reviews.append("⚠️ No recycling selected. Start recycling paper & plastic at minimum.")
else:
    reviews.append("✅ Recycling practices in place.")

for r in reviews:
    st.write(r)

# -------------------------------
# RL Recommendation
# -------------------------------
recommended_action, new_emission = None, None
if agent:
    obs = pipeline.named_steps["preprocessor"].transform(feat).astype("float32")
    action_idx, _ = agent.predict(obs[0], deterministic=True)
    recommended_action = ACTIONS[int(action_idx)]

    if recommended_action == "no_change":
        st.success("Your lifestyle looks efficient already — no changes recommended.")
    else:
        st.info(f"🤖 RL recommends: **{recommended_action.replace('_',' ').title()}**")

        # Apply action
        mod = row.copy()
        if recommended_action == "reduce_car_frequency":
            mod["Vehicle Monthly Distance Km"] = max(0, int(mod["Vehicle Monthly Distance Km"] * 0.7))
            if mod["Transport"] == "private":
                mod["Transport"] = "public"
        elif recommended_action == "switch_to_public_transport":
            mod["Transport"] = "public"
        elif recommended_action == "reduce_meat_consumption":
            mod["Diet"] = "vegetarian"
        elif recommended_action == "switch_to_renewable_energy":
            mod["Heating Energy Source"] = "renewable"
            mod["Energy efficiency"] = "high"
        elif recommended_action == "increase_recycling":
            parts = set(mod["Recycling"].split("|")) if mod["Recycling"] else set()
            parts.update(["paper","plastic"])
            mod["Recycling"] = "|".join(sorted(parts))

        feat_mod = build_features(pd.DataFrame([mod]))
        new_emission = pipeline.predict(feat_mod)[0]

        st.subheader("📉 Emission Reduction Estimate")
        st.write(f"After applying **{recommended_action.replace('_',' ').title()}**, "
                 f"emission would be: **{new_emission:.2f} units**")
        st.success(f"Estimated reduction: **{baseline - new_emission:.2f} units**")

        # Plotly bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Baseline", "After Action"],
            y=[baseline, new_emission],
            marker_color=["red","green"],
            text=[f"{baseline:.2f}", f"{new_emission:.2f}"],
            textposition="auto"
        ))
        fig.update_layout(title="Carbon Emission: Before vs After Recommendation",
                          yaxis_title="Carbon Emission (units)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("⚠️ RL agent not available. Train it in the notebook to enable recommendations.")
