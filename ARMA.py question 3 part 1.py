import pandas as pd
import numpy as np
import statsmodels.api as sm

def predict_stock_price(data, order, live_input):

    
    
    
    # Fit the ARMA model to the data
    model = sm.tsa.ARMA(data, order=order).fit()
    
    # Predict the next stock price
    prediction = model.predict(len(data), len(data) + 1, dynamic=False)[-1]
    
    # Combine the prediction with the most recent stock price
    predicted_price = prediction + live_input
    
    return predicted_price
