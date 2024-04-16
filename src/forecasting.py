import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_reader as dr

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from auxiliary import adf_test, performance_analysis, grid_search
import variables as var
from variables import predictionCount
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Non-invertible|Non-stationary"
)


def view_trend_seasonality(series):
    results = seasonal_decompose(series)
    print(results.seasonal)
    results.plot()
    plt.show()


def arima_prediction(series, test_size):
    arima = auto_arima(
        series,
        test="adf",
        start_p=0,
        start_q=0,
        max_p=6,
        max_q=6,
        seasonal=False,
        trace=False,
        errpr_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    # train and test set split (assuming we try to predic half a year)
    train_size = len(series) - test_size

    train = series[:train_size]
    test = series[train_size:]

    start = len(train)
    end = start + len(test) - 1

    model = grid_search(series, arima.order)
    results = model.fit()

    predictions = results.predict(start=start, end=end, typ="levels").rename(var.ARIMAP)
    return (test, predictions)


# Seasonal AutoRegressive Integrated Moving Average Model
def sarima_prediction(series, test_size):
    sarima = auto_arima(
        series,
        m=12,
        seasonal=True,
        d=None,
        test="adf",
        start_p=0,
        start_q=0,
        max_p=12,
        max_q=12,
        D=None,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    order = sarima.order
    seasonal_order = sarima.seasonal_order

    train_size = len(series) - test_size

    train = series[:train_size]
    test = series[train_size:]

    start = len(train)
    end = start + len(test) - 1
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()

    predictions = results.predict(start, end, typ="levels").rename(
        var.SARIMAP + str(order) + "X" + str(seasonal_order)
    )
    predictions.index = test.index
    print(predictions)
    return (test, predictions)


def series2tuple(series):
    return (series[0], series[1], series[2])


def progressive_prediction(df, energy, pred_algo):
    target = df[energy].drop(df[df[energy] == 0].index)
    print(len(target))
    start = int(len(target) - predictionCount)
    arr_mae = []
    arr_mse = []
    arr_rmse = []
    arr_dm = []
    predictions = []
    originals = []
    period = []
    for i in range(start, len(target)):
        if pred_algo == var.SARIMA:
            (a, b) = sarima_prediction(target[:i], predictionCount)
            print(a)
        else:
            (a, b) = arima_prediction(target[:i], predictionCount)
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
    out[var.DATE] = period
    out.set_index(var.DATE, inplace=True)
    return out


def generate_csv(series, country, energy):
    series.to_csv(
        var.result_dir + "prediction_" + country + "_" + energy + ".csv",
        encoding="utf-8",
    )


def generate_csv_all(arima_series, sarima_series, country, energy):
    df = pd.DataFrame(index=arima_series.index)
    df[var.ARIMA] = arima_series[var.RMSE]
    df[var.SARIMA] = sarima_series[var.RMSE]
    df[var.MEAN] = sarima_series[var.MEAN]
    df[var.SOURCE] = country + "_" + energy
    df[var.ARIMAP] = arima_series[var.FORECAST]
    df[var.SARIMAP] = sarima_series[var.FORECAST]
    df.to_csv(
        "../results/prediction_" + country + "_" + energy + "_all.csv",
        encoding="utf-8",
    )


# country = "Switzerland"
# l = organize_table(country)
# energy = "ror"
# print(l[energy][:len(l)])
# generate_csv(progressive_prediction(l, energy, var.SARIMA), country, energy)
# (a,b) = sarima_prediction(l[energy][:len(l)], 3)
# visualize_prediction(l[energy], b, "a")
# df = pd.read_csv("../results/prediction_France_demand_all.csv", index_col="period")
# print(df["demand"])
# print(a, b)
