import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_reader as dr

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from auxiliary import adf_test, performance_analysis, grid_search
from data_visualization import visualize_error
import variables as var


def view_trend_seasonality(series):
    results = seasonal_decompose(series)
    print(results.seasonal)
    results.plot()
    plt.show()


def arima_prediction(series, test_size):
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
    predictions = results.predict(start=start, end=end, typ="levels").rename(
        "ARIMA predictions"
    )
    return (test, predictions)


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
    return (test, predictions)


def series2tuple(series):
    return (series[0], series[1], series[2])


def progressive_prediction(df, energy, pred_algo):
    start = int(len(df) * 0.97) + 1
    prediction_size = 3
    arr_mae = []
    arr_mse = []
    arr_rmse = []
    arr_dm = []
    predictions = []
    originals = []
    period = []
    for i in range(start, len(df) + 1):
        if (pred_algo == var.SARIMA):
            (a, b) = sarima_prediction(df[energy][:i], prediction_size)
        else:
            (a, b) = arima_prediction(df[energy][:i], prediction_size)
        (MAE, MSE, RMSE, data_mean) = performance_analysis(a, b)
        predictions.append(series2tuple(b))
        originals.append(series2tuple(a))
        arr_mae.append(MAE)
        arr_dm.append(data_mean)
        arr_mse.append(MSE)
        arr_rmse.append(RMSE)
        period.append(a.index[0])
    out = pd.DataFrame(arr_mae, columns=["MAE"])
    out["MSE"] = arr_mse
    out["RMSE"] = arr_rmse
    out["Mean"] = arr_dm
    out["Data"] = originals
    out["Forecast"] = predictions
    out["period"] = period
    out.set_index("period", inplace=True)
    print(out.index)
    return out


def generate_csv(series, country, energy):
    series.to_csv(
        "../results/prediction_" + country + "_" + energy,
        encoding="utf-8",
    )

def generate_csv_all(arima_series, sarima_series, country, energy):
    df = pd.DataFrame(index=arima_series.index)
    df[var.ARIMA] = arima_series[var.RMSE]
    df[var.SARIMA] = sarima_series[var.RMSE]
    df.to_csv("../results/prediction_" + country + "_" + energy + "_all",
        encoding="utf-8",)

# country = "France"
# energy = "hydro_nops"
# generate_csv(progressive_prediction(country, energy), country, energy)
# df = pd.read_csv("../results/prediction_France_hydro_nops", index_col="period")
# progressive_prediction()
# print(df["demand"])
