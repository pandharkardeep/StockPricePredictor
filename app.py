import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the LSTM model
model = load_model("model1.h5")  # Replace with the actual path to your LSTM model file

# Load historical stock data
df = pd.read_csv("AAPL.csv")
df = df.drop(df.columns[[0]], axis=1)
df1 = df.reset_index()['close']

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Function to predict stock prices
def predict_stock_prices(num_days):
    temp_input = list(df1[-100:].reshape(1, -1)[0])
    output = []

    for i in range(num_days):
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:]).reshape(1, -1)
            x_input = x_input.reshape((1, 100, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            output.extend(yhat.tolist())
        else:
            x_input = np.array(temp_input).reshape((1, 100, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            output.extend(yhat.tolist())

    return scaler.inverse_transform(np.array(output).reshape(-1, 1))

# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # User input for the number of days to predict
    num_days = st.number_input("Enter the number of days to predict:", min_value=1, max_value=200, step=1)

    # Predict stock prices
    if st.button("Predict"):
        predicted_prices = predict_stock_prices(num_days)

        # Display the predicted prices
        st.write(f"Predicted Stock Prices for the Next {num_days} Days:")
        st.line_chart(pd.DataFrame({"Predicted Prices": predicted_prices.flatten()}, index=np.arange(1, num_days + 1)))

if __name__ == "__main__":
    main()
