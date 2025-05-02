# Importing necessary libraries
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException # Core library for building the API / For raising errors
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

# Creating an instance of the FastAPI class
app = FastAPI()

""""
Defining a Pydantic model with name "StockRequest" 
so any data being sent to the API will be JSON object with the name "stock_name" 
which must be a string, It will make the API ONLY receieves the data in the format we want
"""
class StockRequest(BaseModel):
    stock_name: str
    start_date: str
    end_date: str

# Dictionary (key:value) for Mapping stock names to file paths
STOCK_FILE_PATHS = {
    "TSLA": "data/TSLA_data.csv",
    "AAPL": "data/AAPL_data.csv",
    "AMZN": "data/AMZN_data.csv",
    "MSFT": "data/MSFT_data.csv",
}

# Defines an API endpoint
@app.post("/LSTM_Predict")
async def predict(stock_request: StockRequest): # Defining the function that will be called when /LSTM_Predict endpoint is hit with an order
    stock_name = stock_request.stock_name
    try: # Handling any potential errors
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_csv(file_path, parse_dates=["Date"])

        # Debugging
        print("DataFrame Info:")
        df.info()
        print("\nDataFrame Head:")
        print(df.head())

    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid stock name")


# Data extraction and preprocessing
    df = df[(df["Date"] >= pd.to_datetime(stock_request.start_date)) & 
    (df["Date"] <= pd.to_datetime(stock_request.end_date))]
    
    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Not enough data for prediction.")

    data = df.filter(["Close"]) # Filters the data to keep only the "Close" price column

    dataset = data.values # Extracts values from the data and converts it into an array (dataset)

    train_len = int(np.ceil(len(dataset) * 0.8)) # Training data takes 80% of the total data

    # Creating a MinMaxScalar object to transform the data to range between 0 and 1
    # This improves training stability and performance
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:train_len, :] # Splits the scaled data into the training set

    seq_len = 60 # LSTM Model will use 60 previous days closing prices to predict the next day's price
    x_train, y_train = [], []

    # This loop creates the training sequences
    for i in range(seq_len, len(train_data)):
        x_train.append(train_data[i - seq_len : i, 0]) # Appends a sequence of 60 previous closing prices to x_train
        y_train.append(train_data[i, 0]) # Appends the closing price of the next day (this is the target value that the model will try to predict)

    x_train, y_train = np.array(x_train), np.array(y_train) # Converts the x_train and y_train lists into Numpy arrays
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshapes x_train to the 3D shape expected by the LSTM layer (1 at the end because number of features is only 1 (closing price))

# LSTM model

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))

# Model compilation and training

    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    model.fit(x_train, y_train, batch_size=32, epochs=5)


# Preparing the test data

    test_data = scaled_data[train_len - seq_len :, :] # Extracts the test data
    x_test = []
    y_test = dataset[train_len:, :] # Stores the actual closing prices, used later to evaluate model's performance

    # This loop creates the testing sequences (similar to the training sequence loop)
    for i in range(seq_len, len(test_data)):
        x_test.append(test_data[i - seq_len : i, 0]) # Appends a sequence of 60 previous closing prices to x_test

    x_test = np.array(x_test) # Converts x_test list into a Numpy array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # Reshapes x_test to the 3D shape expected by the LSTM layer (1 at the end because number of features is only 1 (closing price))

# Making predictions and returning the results

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # reverse the scaling we did before so we can get the actual predicted prices

    predict_prices = [price[0] for price in predictions.tolist()] # Extracts the predicted prices into a list

    return {"prediction": predict_prices}
