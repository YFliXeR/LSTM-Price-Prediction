# Project description
This project is an implementation of a machine learning model using LSTM to predict stock prices
The front-end of the application is built using Streamlit 
The back-end is built using FastAPI
The model is trained using historical stock data, and the predicted prices are displayed on a graphical user interface.

# Usage
To run the application, first start the FastAPI server:

```uvicorn API.ModelServe:app --reload```

Then in a separate terminal, start the Streamlit app:

```streamlit run app.py```

You should now be able to access the application in your web browser at http://localhost:8501.

# To install all the necessary libraries
"""
pip install -r requirements.txt
"""

