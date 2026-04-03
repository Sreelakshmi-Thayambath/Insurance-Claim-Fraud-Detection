# Insurance Claim Fraud Detection

## Overview
This project focuses on detecting fraudulent insurance claims using machine learning. It analyzes customer, policy, and incident-related data to classify whether a claim is **fraudulent or legitimate**.

Fraud detection is a critical task in the insurance industry, helping reduce financial losses and improve claim verification efficiency.

---

## Objectives
- Identify patterns associated with fraudulent claims  
- Build a machine learning model for classification  
- Improve model interpretability using explainability techniques  
- Deploy the model as an interactive web application  

---

## Dataset
The dataset contains insurance claim records with features such as:
- Customer details (age, months as customer)
- Policy information (premium, deductible, state)
- Incident details (type, severity, collision type)
- Claim details (vehicle, injury, property claims)
- Verification signals (witnesses, police report)

**Target Variable:**
- `fraud_reported` → Fraudulent (1) / Legitimate (0)

---

## Exploratory Data Analysis (EDA)
Key insights:
- Fraudulent claims tend to have **higher claim amounts**
- Claims with **fewer witnesses** are more likely to be fraud
- **Absence of police reports** increases fraud probability
- Certain incident types show higher fraud rates
- Strong correlation among claim-related features

---

## Data Preprocessing
- Handled categorical variables using encoding
- Built a preprocessing pipeline for consistency
- Managed multicollinearity (handled well by tree models)
- Combined numerical and categorical transformations

---

## Model Development
- Used a **tree-based machine learning model**
- Integrated preprocessing and model into a pipeline
- Chosen because:
  - Handles nonlinear relationships
  - Works well with structured data
  - Robust to outliers and multicollinearity

### Threshold Tuning
- Default threshold: `0.5`
- Used threshold: `0.4`

Reason:
To improve fraud detection sensitivity and reduce false negatives.

---

## Model Explainability
- Implemented SHAP (SHapley Additive Explanations)
- Explains individual predictions
- Shows feature contribution to fraud probability
- Visualized using waterfall plots

---

## Deployment
- Built an interactive web application using Streamlit
- Users can:
  - Input claim details
  - Get fraud prediction
  - View explanation of prediction

---

## Output
- Fraud Probability Score  
- Classification Result (Fraud / Legitimate)  
- SHAP Waterfall Plot  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- SHAP  
- Streamlit  
- Matplotlib  
