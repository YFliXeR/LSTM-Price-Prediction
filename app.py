# Importing necessary libraries
import datetime
import os
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st

# Local URL where the backend LSTM Model is hosted
API_URL = "http://127.0.0.1:8000/LSTM_Predict" 

# Limiting how far back or forward the user can select dates
MIN_DATE = datetime.date(2020, 1, 1)
MAX_DATE = datetime.date(2022, 12, 31)

# Folder where the CSV files are stored
DATA_DIR = "data"


# Streamlit UI setup
def main():
    st.title("Stock Price Predictor")

    stock_name = st.selectbox(
        "Please choose stock name", ("AAPL", "TSLA", "AMZN", "MSFT")
    )

    start_date = st.date_input(
        "Start date", min_value=MIN_DATE, max_value=MAX_DATE, value=MIN_DATE
    )
    end_date = st.date_input(
        "End date", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE
    )

    if start_date > end_date:
        st.error("Error: End date must be after start date.")
        return
    
    # Load local data files
    csv_path = os.path.join(DATA_DIR, f"{stock_name}_data.csv")
    if not os.path.exists(csv_path):
        st.error(f"No local data found for {stock_name} at {csv_path}")
        return

    # Converts the "Date" column into the right datetime format and filters the rows only so only the data with the date range we want remains
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]

    # If there's no data left after filtering it shows a warning 
    if df.empty:
        st.warning("No data available in the selected date range.")
        return

    # Plot actual stock prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual Close Price"))
    fig.update_layout(title=f"{stock_name} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # What's happening when the user press the "Predict" button
    if st.button("Predict"):
        payload = {
            "stock_name": stock_name,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        try:
            response = requests.post(API_URL, json=payload) # Sends the payload into the backend model
            response.raise_for_status()

            predictions = response.json()["prediction"] # If the request is successful it retrieves the prediction values
            pred_dates = df["Date"].iloc[-len(predictions):]

            fig = go.Figure() # Creating a new plot comparing actual vs predicted prices
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual"))
            fig.add_trace(go.Scatter(x=pred_dates, y=predictions, name="Predicted"))
            fig.update_layout(
                title=f"{stock_name} - Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price"

            )
            st.plotly_chart(fig) # Shows it in streamlit

        except requests.exceptions.RequestException as e: # If there's an error in the request (server is offline), it will show this error
            st.error(f"Error occurred while making the prediction request:\n\n{e}")

# App entry point that runs the main() function
if __name__ == "__main__":
    
    main()
