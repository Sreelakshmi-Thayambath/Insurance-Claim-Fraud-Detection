# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(
#     page_title="Insurance Claim Fraud Detection",
#     layout="wide"
# )

# st.title("🚨 Insurance Claim Fraud Detection")
# st.write("Predict whether an insurance claim is **fraudulent** using a trained ML model.")

# # -----------------------------
# # Load artifacts
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent

# @st.cache_resource
# def load_artifacts():
#     with open(BASE_DIR / "fraud_model.pkl", "rb") as f:
#         model = pickle.load(f)

#     with open(BASE_DIR / "feature_names.pkl", "rb") as f:
#         feature_names = pickle.load(f)

#     threshold = 0.4  # your tuned threshold
#     return model, feature_names, threshold

# model, feature_names, threshold = load_artifacts()

# # -----------------------------
# # Initialize input dataframe
# # -----------------------------
# input_data = pd.DataFrame(
#     columns=feature_names,
#     data=[{col: None for col in feature_names}]
# )

# # -----------------------------
# # Sidebar inputs
# # -----------------------------
# st.sidebar.header("🧾 Claim Details")

# # ---- Numerical inputs ----
# input_data["months_as_customer"] = st.sidebar.number_input(
#     "Months as Customer", min_value=0, max_value=600, value=120
# )

# input_data["age"] = st.sidebar.number_input(
#     "Age", min_value=18, max_value=100, value=35
# )

# input_data["policy_deductable"] = st.sidebar.selectbox(
#     "Policy Deductible", [500, 1000, 2000]
# )

# input_data["policy_annual_premium"] = st.sidebar.number_input(
#     "Annual Premium", min_value=300.0, max_value=3000.0, value=1200.0
# )

# input_data["number_of_vehicles_involved"] = st.sidebar.selectbox(
#     "Vehicles Involved", [1, 2, 3, 4]
# )

# input_data["bodily_injuries"] = st.sidebar.selectbox(
#     "Bodily Injuries", [0, 1, 2]
# )

# input_data["witnesses"] = st.sidebar.selectbox(
#     "Witnesses", [0, 1, 2, 3]
# )

# input_data["total_claim_amount"] = st.sidebar.number_input(
#     "Total Claim Amount", min_value=0.0, value=50000.0
# )

# input_data["injury_claim"] = st.sidebar.number_input(
#     "Injury Claim Amount", min_value=0.0, value=5000.0
# )

# input_data["property_claim"] = st.sidebar.number_input(
#     "Property Claim Amount", min_value=0.0, value=5000.0
# )

# input_data["vehicle_claim"] = st.sidebar.number_input(
#     "Vehicle Claim Amount", min_value=0.0, value=30000.0
# )

# # ---- Categorical inputs ----
# input_data["policy_state"] = st.sidebar.selectbox(
#     "Policy State", ["OH", "IL", "IN"]
# )

# input_data["policy_csl"] = st.sidebar.selectbox(
#     "Policy CSL", ["100/300", "250/500", "500/1000"]
# )

# input_data["insured_sex"] = st.sidebar.selectbox(
#     "Insured Sex", ["MALE", "FEMALE"]
# )

# input_data["insured_education_level"] = st.sidebar.selectbox(
#     "Education Level",
#     ["High School", "College", "Associate", "Masters", "PhD", "JD", "MD"]
# )

# input_data["insured_occupation"] = st.sidebar.selectbox(
#     "Occupation",
#     [
#         "tech-support", "sales", "exec-managerial", "craft-repair",
#         "machine-op-inspct", "other-service", "armed-forces",
#         "priv-house-serv", "protective-serv", "handlers-cleaners",
#         "transport-moving", "adm-clerical", "farming-fishing",
#         "prof-specialty"
#     ]
# )

# input_data["insured_relationship"] = st.sidebar.selectbox(
#     "Relationship",
#     ["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"]
# )

# input_data["incident_type"] = st.sidebar.selectbox(
#     "Incident Type",
#     ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
# )

# input_data["collision_type"] = st.sidebar.selectbox(
#     "Collision Type",
#     ["Rear Collision", "Side Collision", "Front Collision"]
# )

# input_data["incident_severity"] = st.sidebar.selectbox(
#     "Incident Severity",
#     ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
# )

# input_data["authorities_contacted"] = st.sidebar.selectbox(
#     "Authorities Contacted",
#     ["Police", "Fire", "Ambulance", "Other"]
# )

# input_data["property_damage"] = st.sidebar.selectbox(
#     "Property Damage", ["YES", "NO"]
# )

# input_data["police_report_available"] = st.sidebar.selectbox(
#     "Police Report Available", ["YES", "NO"]
# )

# input_data["has_umbrella"] = st.sidebar.selectbox(
#     "Umbrella Policy", [0, 1]
# )

# # -----------------------------
# # Safety check
# # -----------------------------
# missing = set(feature_names) - set(input_data.columns)
# if missing:
#     st.error(f"Missing features: {missing}")
#     st.stop()

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("🔍 Predict Fraud"):
#     prob = model.predict_proba(input_data)[0][1]
#     prediction = int(prob >= threshold)

#     st.subheader("📊 Prediction Result")

#     st.metric("Fraud Probability", f"{prob:.2%}")

#     if prediction == 1:
#         st.error("🚨 Fraudulent Claim Detected")
#     else:
#         st.success("✅ Legitimate Claim")

#     st.caption(f"Decision Threshold: {threshold}")









# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(
#     page_title="Insurance Claim Fraud Detection",
#     layout="wide"
# )

# st.title("🚨 Insurance Claim Fraud Detection")
# st.write("Predict whether an insurance claim is **fraudulent** using a trained ML model.")

# # -----------------------------
# # Base directory
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "model"

# # -----------------------------
# # Load model & artifacts
# # -----------------------------
# @st.cache_resource
# def load_artifacts():
#     model_path = MODEL_DIR / "fraud_model.pkl"
#     features_path = MODEL_DIR / "feature_names.pkl"

#     if not model_path.exists() or not features_path.exists():
#         st.error(f"Model or feature file not found in {MODEL_DIR}. Please make sure 'fraud_model.pkl' "
#                  f"and 'feature_names.pkl' exist.")
#         st.stop()

#     artifacts = joblib.load(model_path)
#     model = artifacts["model"] if isinstance(artifacts, dict) else artifacts
#     threshold = artifacts.get("threshold", 0.4) if isinstance(artifacts, dict) else 0.4

#     with open(features_path, "rb") as f:
#         feature_names = joblib.load(f)

#     return model, feature_names, threshold

# model, feature_names, threshold = load_artifacts()

# # -----------------------------
# # Initialize input dataframe
# # -----------------------------
# input_data = pd.DataFrame(
#     columns=feature_names,
#     data=[{col: None for col in feature_names}]
# )

# # -----------------------------
# # Sidebar inputs
# # -----------------------------
# st.sidebar.header("🧾 Claim Details")

# # Numerical inputs
# input_data["months_as_customer"] = st.sidebar.number_input("Months as Customer", min_value=0, max_value=600, value=120)
# input_data["age"] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
# input_data["policy_deductable"] = st.sidebar.selectbox("Policy Deductible", [500, 1000, 2000])
# input_data["policy_annual_premium"] = st.sidebar.number_input("Annual Premium", min_value=300.0, max_value=3000.0, value=1200.0)
# input_data["number_of_vehicles_involved"] = st.sidebar.selectbox("Vehicles Involved", [1, 2, 3, 4])
# input_data["bodily_injuries"] = st.sidebar.selectbox("Bodily Injuries", [0, 1, 2])
# input_data["witnesses"] = st.sidebar.selectbox("Witnesses", [0, 1, 2, 3])
# input_data["total_claim_amount"] = st.sidebar.number_input("Total Claim Amount", min_value=0.0, value=50000.0)
# input_data["injury_claim"] = st.sidebar.number_input("Injury Claim Amount", min_value=0.0, value=5000.0)
# input_data["property_claim"] = st.sidebar.number_input("Property Claim Amount", min_value=0.0, value=5000.0)
# input_data["vehicle_claim"] = st.sidebar.number_input("Vehicle Claim Amount", min_value=0.0, value=30000.0)

# # Categorical inputs
# input_data["policy_state"] = st.sidebar.selectbox("Policy State", ["OH", "IL", "IN"])
# input_data["policy_csl"] = st.sidebar.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
# input_data["insured_sex"] = st.sidebar.selectbox("Insured Sex", ["MALE", "FEMALE"])
# input_data["insured_education_level"] = st.sidebar.selectbox(
#     "Education Level", ["High School", "College", "Associate", "Masters", "PhD", "JD", "MD"]
# )
# input_data["insured_occupation"] = st.sidebar.selectbox(
#     "Occupation",
#     ["tech-support", "sales", "exec-managerial", "craft-repair",
#      "machine-op-inspct", "other-service", "armed-forces",
#      "priv-house-serv", "protective-serv", "handlers-cleaners",
#      "transport-moving", "adm-clerical", "farming-fishing", "prof-specialty"]
# )
# input_data["insured_relationship"] = st.sidebar.selectbox(
#     "Relationship", ["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"]
# )
# input_data["incident_type"] = st.sidebar.selectbox(
#     "Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
# )
# input_data["collision_type"] = st.sidebar.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision"])
# input_data["incident_severity"] = st.sidebar.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"])
# input_data["authorities_contacted"] = st.sidebar.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other"])
# input_data["property_damage"] = st.sidebar.selectbox("Property Damage", ["YES", "NO"])
# input_data["police_report_available"] = st.sidebar.selectbox("Police Report Available", ["YES", "NO"])
# input_data["has_umbrella"] = st.sidebar.selectbox("Umbrella Policy", [0, 1])

# # -----------------------------
# # Safety check
# # -----------------------------
# missing = set(feature_names) - set(input_data.columns)
# if missing:
#     st.error(f"Missing features: {missing}")
#     st.stop()

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("🔍 Predict Fraud"):
#     prob = model.predict_proba(input_data)[0][1]
#     prediction = int(prob >= threshold)

#     st.subheader("📊 Prediction Result")
#     st.metric("Fraud Probability", f"{prob:.2%}")

#     if prediction == 1:
#         st.error("🚨 Fraudulent Claim Detected")
#     else:
#         st.success("✅ Legitimate Claim")

#     st.caption(f"Decision Threshold: {threshold}")










# import streamlit as st
# import joblib
# import pandas as pd
# from pathlib import Path

# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(
#     page_title="Insurance Claim Fraud Detection",
#     layout="wide"
# )

# st.title("🚨 Insurance Claim Fraud Detection")
# st.write("Predict whether an insurance claim is **fraudulent** using a trained ML model.")

# # -----------------------------
# # Paths
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "model"

# # -----------------------------
# # Load model (UPDATED)
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model_path = MODEL_DIR / "fraud_model.pkl"

#     if not model_path.exists():
#         st.error(f"Missing model file in {MODEL_DIR}")
#         st.stop()

#     model = joblib.load(model_path)
#     threshold = 0.4

#     return model, threshold


# model, threshold = load_model()

# # -----------------------------
# # Sidebar inputs
# # -----------------------------
# st.sidebar.header("🧾 Claim Details")

# months_as_customer = st.sidebar.number_input("Months as Customer", 0, 600, 120)
# age = st.sidebar.number_input("Age", 18, 100, 35)
# policy_state = st.sidebar.selectbox("Policy State", ["OH", "IL", "IN"])
# policy_csl = st.sidebar.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
# policy_deductable = st.sidebar.selectbox("Policy Deductible", [500, 1000, 2000])
# policy_annual_premium = st.sidebar.number_input("Annual Premium", 300.0, 3000.0, 1200.0)
# insured_sex = st.sidebar.selectbox("Insured Sex", ["MALE", "FEMALE"])
# insured_education_level = st.sidebar.selectbox(
#     "Education Level",
#     ["High School", "College", "Associate", "Masters", "PhD", "JD", "MD"]
# )
# insured_occupation = st.sidebar.selectbox(
#     "Occupation",
#     [
#         "tech-support", "sales", "exec-managerial", "craft-repair",
#         "machine-op-inspct", "other-service", "armed-forces",
#         "priv-house-serv", "protective-serv", "handlers-cleaners",
#         "transport-moving", "adm-clerical", "farming-fishing",
#         "prof-specialty"
#     ]
# )
# insured_relationship = st.sidebar.selectbox(
#     "Relationship",
#     ["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"]
# )

# incident_type = st.sidebar.selectbox(
#     "Incident Type",
#     ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
# )
# collision_type = st.sidebar.selectbox(
#     "Collision Type",
#     ["Rear Collision", "Side Collision", "Front Collision"]
# )
# incident_severity = st.sidebar.selectbox(
#     "Incident Severity",
#     ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
# )
# authorities_contacted = st.sidebar.selectbox(
#     "Authorities Contacted",
#     ["Police", "Fire", "Ambulance", "Other"]
# )

# incident_hour_of_the_day = st.sidebar.slider("Incident Hour of the Day", 0, 23, 12)
# number_of_vehicles_involved = st.sidebar.selectbox("Vehicles Involved", [1, 2, 3, 4])
# property_damage = st.sidebar.selectbox("Property Damage", ["YES", "NO"])
# bodily_injuries = st.sidebar.selectbox("Bodily Injuries", [0, 1, 2])
# witnesses = st.sidebar.selectbox("Witnesses", [0, 1, 2, 3])
# police_report_available = st.sidebar.selectbox("Police Report Available", ["YES", "NO"])

# total_claim_amount = st.sidebar.number_input("Total Claim Amount", 0.0, 100000.0, 50000.0)
# injury_claim = st.sidebar.number_input("Injury Claim", 0.0, 50000.0, 5000.0)
# property_claim = st.sidebar.number_input("Property Claim", 0.0, 50000.0, 5000.0)
# vehicle_claim = st.sidebar.number_input("Vehicle Claim", 0.0, 80000.0, 30000.0)
# has_umbrella = st.sidebar.selectbox("Umbrella Policy", [0, 1])

# # -----------------------------
# # Build input dataframe (RAW)
# # -----------------------------
# input_data = pd.DataFrame([{
#     "months_as_customer": months_as_customer,
#     "age": age,
#     "policy_state": policy_state,
#     "policy_csl": policy_csl,
#     "policy_deductable": policy_deductable,
#     "policy_annual_premium": policy_annual_premium,
#     "insured_sex": insured_sex,
#     "insured_education_level": insured_education_level,
#     "insured_occupation": insured_occupation,
#     "insured_relationship": insured_relationship,
#     "incident_type": incident_type,
#     "collision_type": collision_type,
#     "incident_severity": incident_severity,
#     "authorities_contacted": authorities_contacted,
#     "incident_hour_of_the_day": incident_hour_of_the_day,
#     "number_of_vehicles_involved": number_of_vehicles_involved,
#     "property_damage": property_damage,
#     "bodily_injuries": bodily_injuries,
#     "witnesses": witnesses,
#     "police_report_available": police_report_available,
#     "total_claim_amount": total_claim_amount,
#     "injury_claim": injury_claim,
#     "property_claim": property_claim,
#     "vehicle_claim": vehicle_claim,
#     "has_umbrella": has_umbrella
# }])

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("🔍 Predict Fraud"):
#     prob = model.predict_proba(input_data)[0, 1]
#     prediction = int(prob >= threshold)

#     st.subheader("📊 Prediction Result")
#     st.metric("Fraud Probability", f"{prob:.2%}")

#     if prediction:
#         st.error("🚨 Fraudulent Claim Detected")
#     else:
#         st.success("✅ Legitimate Claim")

#     st.caption(f"Decision Threshold: {threshold}")





import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    layout="wide"
)

st.title("🚨 Insurance Claim Fraud Detection")
st.write("Predict whether an insurance claim is **fraudulent** using a trained ML model.")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# -----------------------------
# Load model & feature names
# -----------------------------
@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "fraud_model.pkl"
    features_path = MODEL_DIR / "feature_names.pkl"

    if not model_path.exists() or not features_path.exists():
        st.error(f"Missing model files in {MODEL_DIR}")
        st.stop()

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    threshold = 0.4

    return model, feature_names, threshold


model, feature_names, threshold = load_model()

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("🧾 Claim Details")

months_as_customer = st.sidebar.number_input("Months as Customer", 0, 600, 120)
age = st.sidebar.number_input("Age", 18, 100, 35)
policy_state = st.sidebar.selectbox("Policy State", ["OH", "IL", "IN"])
policy_csl = st.sidebar.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
policy_deductable = st.sidebar.selectbox("Policy Deductible", [500, 1000, 2000])
policy_annual_premium = st.sidebar.number_input("Annual Premium", 300.0, 3000.0, 1200.0)
insured_sex = st.sidebar.selectbox("Insured Sex", ["MALE", "FEMALE"])
insured_education_level = st.sidebar.selectbox(
    "Education Level",
    ["High School", "College", "Associate", "Masters", "PhD", "JD", "MD"]
)
insured_occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "tech-support", "sales", "exec-managerial", "craft-repair",
        "machine-op-inspct", "other-service", "armed-forces",
        "priv-house-serv", "protective-serv", "handlers-cleaners",
        "transport-moving", "adm-clerical", "farming-fishing",
        "prof-specialty"
    ]
)
insured_relationship = st.sidebar.selectbox(
    "Relationship",
    ["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"]
)

incident_type = st.sidebar.selectbox(
    "Incident Type",
    ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
)
collision_type = st.sidebar.selectbox(
    "Collision Type",
    ["Rear Collision", "Side Collision", "Front Collision"]
)
incident_severity = st.sidebar.selectbox(
    "Incident Severity",
    ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
)
authorities_contacted = st.sidebar.selectbox(
    "Authorities Contacted",
    ["Police", "Fire", "Ambulance", "Other"]
)

incident_hour_of_the_day = st.sidebar.slider("Incident Hour of the Day", 0, 23, 12)
number_of_vehicles_involved = st.sidebar.selectbox("Vehicles Involved", [1, 2, 3, 4])
property_damage = st.sidebar.selectbox("Property Damage", ["YES", "NO"])
bodily_injuries = st.sidebar.selectbox("Bodily Injuries", [0, 1, 2])
witnesses = st.sidebar.selectbox("Witnesses", [0, 1, 2, 3])
police_report_available = st.sidebar.selectbox("Police Report Available", ["YES", "NO"])

total_claim_amount = st.sidebar.number_input("Total Claim Amount", 0.0, 100000.0, 50000.0)
injury_claim = st.sidebar.number_input("Injury Claim", 0.0, 50000.0, 5000.0)
property_claim = st.sidebar.number_input("Property Claim", 0.0, 50000.0, 5000.0)
vehicle_claim = st.sidebar.number_input("Vehicle Claim", 0.0, 80000.0, 30000.0)
has_umbrella = st.sidebar.selectbox("Umbrella Policy", [0, 1])

# -----------------------------
# Build input dataframe (RAW)
# -----------------------------
input_data = pd.DataFrame([{
    "months_as_customer": months_as_customer,
    "age": age,
    "policy_state": policy_state,
    "policy_csl": policy_csl,
    "policy_deductable": policy_deductable,
    "policy_annual_premium": policy_annual_premium,
    "insured_sex": insured_sex,
    "insured_education_level": insured_education_level,
    "insured_occupation": insured_occupation,
    "insured_relationship": insured_relationship,
    "incident_type": incident_type,
    "collision_type": collision_type,
    "incident_severity": incident_severity,
    "authorities_contacted": authorities_contacted,
    "incident_hour_of_the_day": incident_hour_of_the_day,
    "number_of_vehicles_involved": number_of_vehicles_involved,
    "property_damage": property_damage,
    "bodily_injuries": bodily_injuries,
    "witnesses": witnesses,
    "police_report_available": police_report_available,
    "total_claim_amount": total_claim_amount,
    "injury_claim": injury_claim,
    "property_claim": property_claim,
    "vehicle_claim": vehicle_claim,
    "has_umbrella": has_umbrella
}])

# -----------------------------
# Prediction + SHAP
# -----------------------------
if st.button("🔍 Predict Fraud"):

    # Prediction
    prob = model.predict_proba(input_data)[0, 1]
    prediction = int(prob >= threshold)

    st.subheader("📊 Prediction Result")
    st.metric("Fraud Probability", f"{prob:.2%}")

    if prediction:
        st.error("🚨 Fraudulent Claim Detected")
    else:
        st.success("✅ Legitimate Claim")

    st.caption(f"Decision Threshold: {threshold}")

# -----------------------------
# SHAP Explanation
# -----------------------------
    st.subheader("🧠 SHAP Explanation (Why this prediction?)")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

# Transform input (NO SMOTE)
    input_processed = preprocessor.transform(input_data)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_processed)

# -----------------------------
# Handle SHAP output correctly
# -----------------------------

# Case 1: SHAP returns list (older versions)
    if isinstance(shap_values, list):
        shap_values_sample = shap_values[1][0]      # class 1, first sample
        expected_value = explainer.expected_value[1]

# Case 2: SHAP returns array (newer versions)
    else:
    # shap_values shape could be (n_samples, n_features) OR (n_samples, n_features, n_classes)

        if shap_values.ndim == 3:
            shap_values_sample = shap_values[0, :, 1]   # sample 0, fraud class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_sample = shap_values[0]          # already single output
            expected_value = explainer.expected_value

# -----------------------------
# Waterfall plot (SINGLE sample)
# -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_sample,
            base_values=expected_value,
            data=input_processed[0],
            feature_names=feature_names
        ),
        show=False
    )

    st.pyplot(fig)

