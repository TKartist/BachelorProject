import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_reader as dr
import itertools

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from auxiliary import adf_test, performance_analysis


def view_trend_seasonality(series):
    results = seasonal_decompose(series)
    print(results.seasonal)
    results.plot()
    plt.show()


# there are obscure cases of auto_arima where the minimum AIC is present after an inverted bell curve
# hence, in this case, we perform a grid search just in case.
# This function will be alter the order, if it wasn't intercepted by auto_arima function as it should
# i.e. (p, d, q) == (0,0,0)
def grid_search(series, order):
    if order[0] > 0 or order[2] > 0:
        # perform grid_search
        model = ARIMA(series, order=order)
        return model
    p = range(1, 7)
    q = range(1, 7)
    best_aic = np.inf
    best_model = None
    pdq = list(itertools.product(p, order, q))
    for param in pdq:
        try:
            model = ARIMA(series, order=param)
            results = model.fit()
            aic = results.aic

            if results.mle_retvals["converged"] and aic < best_aic:
                best_model = model
                best_aic = aic
        except:
            continue
    return best_model


def arima_order(series, test_size):
    arima = auto_arima(
        series,
        start_p=0,
        start_q=0,
        max_p=6,
        max_q=6,
        seasonal=False,
        trace=True,
        errpr_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    # train and test set split (assuming we try to predic half a year)
    train = series[:-test_size]
    test = series[-test_size:]

    model = grid_search(series, (3, 0, 1))
    results = model.fit()

    start = len(train)
    end = len(train) + len(test) - 1
    preds = results.predict(start=start, end=end, typ="levels").rename(
        "ARIMA predictions"
    )
    print(train)
    print(preds)
    print(test)
    series.plot(legend=True, figsize=(16, 10))
    preds.plot(legend=True)
    plt.show()


# Seasonal AutoRegressive Integrated Moving Average Model
def sarima_prediction(series, test_size):
    sarima = auto_arima(
        series,
        start_p=0,
        start_q=0,
        max_p=10,
        max_q=10,
        seasonal=True,
        m=12,
    )
    order = sarima.order
    seasonal_order = sarima.seasonal_order

    train = series[:-test_size]
    test = series[-test_size:]

    start = len(train)
    end = start + len(test) - 1

    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order, enforce_invertibility=False
    )
    results = model.fit()
    predictions = results.predict(start, end, typ="levels").rename(
        "SARIMA PREDICTION " + str(order) + "X" + str(seasonal_order)
    )
    # test.plot(legend=True, figsize=(16, 10))
    # predictions.plot(legend=True)
    # plt.show()
    return (test, predictions)


# df = dr.organize_table("France")
# view_trend_seasonality(df["hydro_nops"])
# arima_order(df["hydro_nops"], 6)
# (a, b) = sarima_prediction(df["hydro_nops"], 3)
# analysis = performance_analysis(b, a)
# print(analysis)


def progressive_prediction():
    df = dr.organize_table("France")
    start = int(len(df) * 0.6) + 1
    prediction_size = 3
    arr_mae = []
    arr_mse = []
    arr_rmse = []
    arr_dm = []
    predictions = []
    originals = []
    for i in range(start, len(df) + 1):
        (a, b) = sarima_prediction(df["hydro_nops"][:i], prediction_size)
        (MAE, MSE, RMSE, data_mean) = performance_analysis(a, b)
        predictions.append(b)
        originals.append(a)
        arr_mae.append(MAE)
        arr_dm.append(data_mean)
        arr_mse.append(MSE)
        arr_rmse.append(RMSE)
    out = pd.DataFrame(arr_mae, columns=["MAE"])
    out["MSE"] = arr_mse
    out["RMSE"] = arr_rmse
    out["Mean"] = arr_dm
    out["Data"] = originals
    out["Forecast"] = predictions
    print(out)
    return out


progressive_prediction()
# progressive_prediction()
# print(df["demand"])
