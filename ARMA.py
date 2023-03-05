import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA

# Load the data
data = pd.read_csv('your_data_file.csv', index_col='Date', parse_dates=True)

# Split the data into training and testing sets
train_data = data[:'2022-01-31']
test_data = data['2022-02-01':]

# Create the ARMA model
model = ARMA(train_data, order=(2, 1))

# Fit the model to the training data
results = model.fit()

# Generate predictions for the testing set
predictions = results.predict(start='2022-02-01', end='2022-02-28')

# Plot the actual prices and the predicted prices
plt.plot(test_data, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()
