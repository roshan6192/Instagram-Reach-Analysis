import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("📊 Instagram Reach Analysis")

# Load Data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data)

    # Donut Chart
    st.subheader("📈 Donut Chart: Impressions by Source")
    long_df = data[['From Home', 'From Hashtags', 'From Explore', 'From Other']].melt(
        var_name='Source', value_name='Impressions'
    )
    summary = long_df.groupby('Source', as_index=False).sum()

    fig = px.pie(summary, values='Impressions', names='Source',
                 title='Impressions from Various Sources', hole=0.5)
    st.plotly_chart(fig)

    # Correlation
    st.subheader("📌 Correlation with Impressions")
    numeric_data = data.select_dtypes(include='number')
    correlation = numeric_data.corr()
    st.dataframe(correlation["Impressions"].sort_values(ascending=False))
