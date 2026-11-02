# Importing necessary packages

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Page Configuration

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("Retail Sales Forecasting")
st.write("Forecast future sales using Linear Regression.")

# Load saved model and last date
model = joblib.load("model.joblib")
last_date = pd.to_datetime(joblib.load("last_date.joblib"))

# User input
n = st.number_input("How many days do you want to forecast?",min_value = 1,max_value = 365, value = 7)

# Predict sales
if st.button("Predict Sales"):

    dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=int(n)
    )

    days = pd.DataFrame({
        "day": dates.map(pd.Timestamp.toordinal)
    })

    predictions = model.predict(days)

    # Display results
    result = pd.DataFrame({
        "Date": dates,
        "Predicted Sales": predictions
    })

    st.subheader("Forecast Results")
    st.dataframe(result, hide_index=True)

    # Display graph
    fig, ax = plt.subplots()
    ax.plot(dates, predictions, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Future Sales Forecast")
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)