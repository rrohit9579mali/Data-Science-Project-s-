import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Page Setup ---
st.set_page_config(page_title="Water Quality Predictor", layout="centered")

# Custom CSS for better look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: ; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Title ---
st.title('💧 Water Quality Intelligence')
st.markdown("Enter sensor values to check if the water is safe for use.")

# --- Data Loading & Model Training ---
@st.cache_resource
def get_trained_model():
    # Loading the file
    # Note: Make sure the path matches your GitHub structure (usually just 'water_quality_class.csv')
    data = pd.read_csv('pythonProject/water_quality_class.csv')
    st.dataframe(data.head())
    # Features (X) and Target (y)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model with Best Params
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(x_train, y_train)

    # Accuracy Calculation
    acc = accuracy_score(y_test, model.predict(x_test))
    return model, acc, x.columns.tolist()

# Load model and accuracy
try:
    model, accuracy, feature_names = get_trained_model()

    # --- 1. Display Accuracy on TOP ---
    st.metric(label="✅ Model Prediction Accuracy", value=f"{accuracy * 100:.2f}%")
    st.divider()

    # --- 2. Prediction Inputs ---
    st.subheader("🔍 Real-time Prediction")

    col1, col2 = st.columns(2)

    with col1:
        val1 = st.number_input(f"Enter {feature_names[0]}", min_value=0.0, step=0.1)
    with col2:
        val2 = st.number_input(f"Enter {feature_names[1]}", min_value=0.0, step=0.1)

    # --- 3. Prediction Button & Logic ---
    if st.button('Check Quality'):
        # Model prediction
        prediction_value = model.predict([[val1, val2]])[0]

        # Mapping: 0=Poor, 1=Moderate, 2=Good
        result_map = {0: "Poor ❌", 1: "Moderate ⚠️", 2: "Good ✅"}

        # Displaying result
        res_text = result_map.get(prediction_value, str(prediction_value))

        st.success(f"### Predicted Quality: {res_text}")

        # Extra feedback based on result in English
        if prediction_value == 2:
            st.info("The water is safe for drinking!")
        elif prediction_value == 1:
            st.warning("The water is acceptable but filtration is recommended.")
        elif prediction_value == 0:
            st.error("Warning: This water is unsafe for consumption!")

except Exception as e:
    st.error(
        f"Error: Problem loading the dataset. Please ensure 'water_quality_class.csv' is in the correct path. Details: {e}")

# --- Footer ---
st.sidebar.markdown("### Model Info")
st.sidebar.write("Using Random Forest Classifier with optimized hyperparameters.")
