import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


# ---------- Load model + columns (cached so they load only once) ----------
@st.cache_resource
def load_model():
    model = joblib.load("models/house_price_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    return model, columns


@st.cache_data
def load_data():
    return pd.read_csv("data/Housing.csv")


model, model_columns = load_model()
df = load_data()

# ---------- Header ----------
st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house and get an estimated price, "
    "based on a model trained on real housing data."
)

# ---------- Layout: inputs on the left, result on the right ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("House details")

    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000,
                               value=5000, step=100)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
        stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
        parking = st.number_input("Parking spots", min_value=0, max_value=5, value=1)
    with c2:
        mainroad = st.selectbox("On a main road?", ["yes", "no"])
        guestroom = st.selectbox("Guest room?", ["yes", "no"])
        basement = st.selectbox("Basement?", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot water heating?", ["yes", "no"])
        airconditioning = st.selectbox("Air conditioning?", ["yes", "no"])
        prefarea = st.selectbox("Preferred area?", ["yes", "no"])

    furnishing = st.selectbox("Furnishing status",
                              ["furnished", "semi-furnished", "unfurnished"])


# ---------- Build a single-row input matching the model's columns ----------
def build_input():
    row = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": 1 if mainroad == "yes" else 0,
        "guestroom": 1 if guestroom == "yes" else 0,
        "basement": 1 if basement == "yes" else 0,
        "hotwaterheating": 1 if hotwaterheating == "yes" else 0,
        "airconditioning": 1 if airconditioning == "yes" else 0,
        "prefarea": 1 if prefarea == "yes" else 0,
    }
    input_df = pd.DataFrame([row])

    # One-hot encode furnishing: set the matching dummy column if it exists.
    # "furnished" is the dropped baseline, so it correctly sets nothing.
    furnish_col = f"furnishingstatus_{furnishing}"
    if furnish_col in model_columns:
        input_df[furnish_col] = 1

    # Reorder and fill to exactly match what the model was trained on.
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    return input_df


with right:
    st.subheader("Estimated price")
    if st.button("Predict price", type="primary"):
        X_new = build_input()
        prediction = model.predict(X_new)[0]
        st.metric(label="Predicted price", value=f"{prediction:,.0f}")
        st.caption("Estimate only — based on patterns in the training data.")
    else:
        st.info("Fill in the details and click **Predict price**.")

# ---------- Data overview ----------
st.divider()
with st.expander("📊 Explore the data this model learned from"):
    st.write(f"The model was trained on {len(df)} houses.")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Price vs Area")
    st.scatter_chart(df, x="area", y="price")