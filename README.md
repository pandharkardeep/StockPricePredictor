# Stock Price Predictor

This project is a Stock Price Predictor based on LSTM (Long Short-Term Memory) neural networks. It utilizes Tensorflow, Keras, and Streamlit for the frontend.

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/pandharkardeep/StockPricePredictor.git
    ```

2. Change into the project directory:

    ```bash
    cd StockPricePredictor
    ```

3. Install the required dependencies. It's recommended to use a virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

This will launch a local server, and you can view the Stock Price Predictor app in your web browser.

## How to Use

1. Once the app is running, it will prompt you to enter the number of days for stock price prediction.

2. Enter the desired number of days and click the "Predict" button.

3. The app will display a chart with the predicted stock prices for the specified number of days.

## Project Structure

- `app.py`: Streamlit app script.
- `main.ipynb`: Contains the LSTM-based stock price predictor model using Tensorflow and Keras.


## Dependencies

- Tensorflow
- Keras
- Streamlit
- Numpy
- Pandas
- Matplotlib


Feel free to explore and modify the code according to your needs. If you have any questions or suggestions, please feel free to contribute or open an issue.

Happy predicting!

