import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("📊 Instagram Reach Analysis + Impressions Predictor")

uploaded_file = st.file_uploader("Upload Instagram Reach CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, encoding='latin1')
        st.success("✅ File uploaded successfully!")

        st.subheader("🔍 Raw Data")
        st.dataframe(data)

        # Pie chart of Impressions Sources
        st.subheader("📈 Impressions From Various Sources")
        sources = ["From Home", "From Hashtags", "From Explore", "From Other"]
        if all(source in data.columns for source in sources):
            summary = data[sources].sum().reset_index()
            summary.columns = ["Source", "Impressions"]
            fig = px.pie(summary, values="Impressions", names="Source", hole=0.5,
                         title="Impressions from Different Sources")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("📊 Correlation with Impressions")
        numeric_data = data.select_dtypes(include=['number'])
        corr = numeric_data.corr()
        st.dataframe(corr["Impressions"].sort_values(ascending=False))

        # Prediction Section
        st.subheader("📌 Predict Impressions")

        features = ["Reach", "Saves", "Likes", "Shares", "Comments",
                    "From Home", "From Hashtags", "From Explore", "From Other"]

        if all(col in data.columns for col in features):
            X = data[features]
            y = data["Impressions"]

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            st.markdown("### 🔢 Enter values to predict Impressions")
            input_data = {}
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0, step=1)

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"🎯 Predicted Impressions: **{int(prediction):,}**")

        else:
            st.warning("Some required columns for prediction are missing in your CSV file.")

    except Exception as e:
        st.error(f"❌ Error reading or processing the file: {e}")
else:
    st.info("📤 Please upload a CSV file to get started.")
