import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction")
st.write("Provide property details and get a price prediction from the saved linear regression model.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_regression_model.pkl")
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "Housing.csv"),
    r"Housing.csv",
]

@st.cache_data
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_training_data():
    for p in DATA_PATHS:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p)
                return df
        except Exception:
            continue
    return None

@st.cache_data
def compute_preprocessors(train_df: pd.DataFrame):
    """Compute area bounds and ordinal encoder from training data."""
    preprocessors = {}
    
    # Compute area bounds (IQR-based capping)
    Q1 = train_df['area'].quantile(0.25)
    Q3 = train_df['area'].quantile(0.75)
    IQR = Q3 - Q1
    preprocessors['area_lower'] = Q1 - 1.5 * IQR
    preprocessors['area_upper'] = Q3 + 1.5 * IQR
    
    # Fit ordinal encoder for furnishingstatus
    oe = OrdinalEncoder(categories=[['unfurnished', 'semi-furnished', 'furnished']])
    oe.fit(train_df[['furnishingstatus']])
    preprocessors['ordinal_encoder'] = oe
    
    return preprocessors

model = load_model()
train_df = load_training_data()

if model is None:
    st.error("Saved model `linear_regression_model.pkl` not found in the app directory. Place the file next to this script.")
    st.stop()

if train_df is None:
    st.error("Training data `Housing.csv` not found. Place it next to this script.")
    st.stop()

preprocessors = compute_preprocessors(train_df)

# Input fields
st.header("Input Features")
area = st.number_input("Area (sq units)", min_value=1.0, value=1000.0, step=1.0)
bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)
stories = st.number_input("Stories", min_value=1, value=1, step=1)
parking = st.number_input("Parking", min_value=0, value=1, step=1)

mainroad = st.selectbox("Main Road?", options=["yes", "no"], index=1)
guestroom = st.selectbox("Guest Room?", options=["yes", "no"], index=1)
basement = st.selectbox("Basement?", options=["yes", "no"], index=1)
hotwaterheating = st.selectbox("Hot Water Heating?", options=["yes", "no"], index=1)
airconditioning = st.selectbox("Air Conditioning?", options=["yes", "no"], index=1)

furnishingstatus = st.selectbox("Furnishing Status", options=["unfurnished", "semi-furnished", "furnished"], index=0)

prefarea = st.selectbox("Preferred Area?", options=["yes", "no"], index=1)

# Preprocessing helper

def preprocess_input(values: dict, preprocessors: dict, train_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([values])

    # Map yes/no to 1/0 for the binary columns
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Ordinal encoding for furnishingstatus using the fitted encoder
    if "furnishingstatus" in df.columns:
        df["furnishingstatus"] = preprocessors['ordinal_encoder'].transform(df[["furnishingstatus"]])

    # Area capping using precomputed bounds
    df['area'] = np.where(
        df['area'] > preprocessors['area_upper'],
        preprocessors['area_upper'],
        np.where(
            df['area'] < preprocessors['area_lower'],
            preprocessors['area_lower'],
            df['area']
        )
    )

    # Ensure column order matches training features
    feature_cols = list(train_df.drop(columns=['price']).columns)

    # Add any missing columns with default 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    df = df[feature_cols]
    return df

# Collect values
values = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'furnishingstatus': furnishingstatus,
    'prefarea': prefarea
}

input_df = preprocess_input(values, preprocessors, train_df)

st.subheader("Preprocessed input")
st.dataframe(input_df)

if st.button("Predict Price"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Predicted price: {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Make sure the model and (optionally) the training CSV are available next to this script.")

st.markdown("---")
st.write("Notes:")
st.write("- The app loads the model `linear_regression_model.pkl` and training data `Housing.csv` at startup.")
st.write("- Preprocessors (area bounds, ordinal encoder) are automatically computed from the training data.")
st.write("- All inputs are preprocessed identically to the notebook before prediction.")
