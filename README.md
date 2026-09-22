# LSTM Stock Price Prediction

This project develops and evaluates Long Short-Term Memory (LSTM) deep learning models for one-step-ahead stock closing price forecasting using historical stock market data from Amazon (AMZN) and Cisco Systems (CSCO).

The project compares a baseline LSTM architecture with an improved LSTM architecture and evaluates their performance using RMSE, MAE, and MAPE.

## Project Objective

The objective of this project is to investigate whether LSTM-based deep learning models can learn temporal patterns from historical stock closing prices and generate one-day-ahead forecasts.

The project follows a time-series forecasting workflow consisting of:

1. Data loading and inspection
2. Data quality checking
3. Exploratory data analysis
4. Feature selection and preprocessing
5. Chronological train-test splitting
6. Data normalization
7. Time-series windowing
8. Train-validation splitting
9. Baseline LSTM development
10. Improved LSTM development
11. Model evaluation and comparison

## Dataset

The project uses historical stock price data for:

- **AMZN** — Amazon
- **CSCO** — Cisco Systems

For the forecasting task, only the following variables are used:

- `Date`
- `Close`

The final **252 trading days**, approximately representing the last year of observations, are reserved as the test set. The remaining observations are used for model development.

## Data Preprocessing

The stock closing prices are normalized using `MinMaxScaler`.

A sliding-window approach is then used to transform the time-series data into supervised learning samples.

### Time-Series Configuration

- **Window size:** 5 trading days
- **Forecast horizon:** 1 trading day
- **Train-validation split:** 90% / 10%
- **Test period:** Last 252 trading days

The model uses the previous five trading days to predict the closing price of the following trading day.

## Baseline LSTM

The baseline model is designed as a simple LSTM architecture to establish a performance benchmark.

### Architecture

- LSTM layer — 50 units
- Activation — ReLU
- Dense output layer — 1 unit
- Optimizer — Adam
- Loss function — Mean Squared Error (MSE)
- Evaluation metric — Mean Absolute Error (MAE)

The same baseline architecture is trained separately for AMZN and CSCO.

## Improved LSTM

The baseline architecture is extended to increase model capacity while introducing regularization.

### Architecture

- LSTM layer — 64 units
- Activation — ReLU
- Dropout — 0.1
- Dense layer — 16 units
- Dense output layer — 1 unit
- Optimizer — Adam
- Loss function — Mean Squared Error (MSE)

Early stopping is also used during training to stop the training process when validation loss no longer improves and to restore the best model weights.

## Model Evaluation

Both the baseline and improved LSTM models are evaluated on the test set using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error

### Results

| Dataset | Model | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|
| AMZN | Baseline LSTM | 61.91 | 45.02 | 2.41% |
| AMZN | Improved LSTM | 50.93 | 35.07 | 1.88% |
| CSCO | Baseline LSTM | 1.05 | 0.71 | 1.55% |
| CSCO | Improved LSTM | 1.45 | 1.11 | 2.31% |

## Key Findings

For **AMZN**, the improved LSTM architecture achieved better performance than the baseline model, reducing RMSE, MAE, and MAPE.

For **CSCO**, the baseline LSTM achieved better performance than the improved architecture across RMSE, MAE, and MAPE.

These results indicate that increasing model complexity does not necessarily improve forecasting performance across different stock datasets. The effectiveness of an architecture depends on the characteristics of the underlying time series.

## Visualization

The project includes visualizations of:

- Historical AMZN closing prices
- Historical CSCO closing prices
- Training and validation learning curves
- Actual vs. predicted AMZN closing prices
- Actual vs. predicted CSCO closing prices

## Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure

```text
LSTM-Stock-Price-Forecasting/
│
├── README.md
├── DeepLearning1-2-3.ipynb
├── AMZN.csv
├── CSCO.csv
└── requirements.txt
