import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_reader as dr

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from auxiliary import adf_test

from statsmodels.tsa.statespace.sarimax import SARIMAX



def view_trend_seasonality(series):
    results = seasonal_decompose(series)
    results.plot()
    plt.show()


def arima_order(series, test_size):
    arima = auto_arima(
        series, start_p=0, start_q=0, max_p=10, max_q=10, seasonal=True, trace=True
    )
    order = arima.order

    # train and test set split (assuming we try to predic half a year)
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]

    model = ARIMA(train, order=order)
    results = model.fit()

    start = len(train)
    end = len(train) + len(test) - 1
    preds = results.predict(start=start, end=end, typ="levels").rename(
        "ARIMA" + str(order) + " predictions"
    )
    print(preds)
    series.plot(legend=True, figsize=(16, 10))
    preds.plot(legend=True)
    plt.show()

def sarimax_prediction(series, test_size):


df = dr.organize_table("Belgium")
# view_trend_seasonality(df["demand"])
arima_order(df["demand"], 12)
# print(df["demand"])
