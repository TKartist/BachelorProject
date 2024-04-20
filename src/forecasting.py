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
    results.plot()
    plt.show()


def arima_prediction(series, test_size):
    arima = auto_arima(
        series,
        test="adf",
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
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
        test="adf",
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        start_P=0,
        start_Q=0,
        max_P=5,
        max_Q=5,
        D=None,
        trace=False,
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
    return (test, predictions)


def series2tuple(series):
    return (series[0], series[1], series[2])


def progressive_prediction(df, energy, pred_algo):
    target = df[energy].drop(df[df[energy] == 0].index)
    start = int(len(target) - predictionCount - 2)
    arr_mae = []
    arr_mse = []
    arr_rmse = []
    arr_dm = []
    period = []
    predictions = []
    pred_col = {}
    for i in range(start, len(target)):
        if pred_algo == var.SARIMA:
            (test, pred) = sarima_prediction(target[:i], predictionCount)
        else:
            (test, pred) = arima_prediction(target[:i], predictionCount)
        (MAE, MSE, RMSE, data_mean) = performance_analysis(test, pred)
        for ind in pred.index:
            if ind in pred_col:
                pred_col[ind].append(pred[ind])
            else:
                pred_col[ind] = [pred[ind]]
        predictions.append(series2tuple(pred))
        arr_mae.append(MAE)
        arr_dm.append(data_mean)
        arr_mse.append(MSE)
        arr_rmse.append(RMSE)
        period.append(test.index[0])
    out = pd.DataFrame(arr_mae, columns=["MAE"])
    out[var.MSE] = arr_mse
    out[var.RMSE] = arr_rmse
    out[var.MEAN] = arr_dm
    out[var.FORECAST] = predictions
    out[var.DATE] = period
    out.set_index(var.DATE, inplace=True)
    return (out, pred_col)


def generate_csv(series, country, energy):
    series.to_csv(
        var.result_dir + "prediction_" + country + "_" + energy + ".csv",
        encoding="utf-8",
    )


def generate_csv_all(sarima_series, arima_series, country, energy):
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


def generate_csv_area_chart(arima_dict, sarima_dict, country, energy):
    df = pd.DataFrame(columns=[var.DATE, var.ARIMAP, var.SARIMAP])
    for key in arima_dict:
        new_row = {
            var.DATE: key,
            var.ARIMAP: arima_dict[key],
            var.SARIMAP: sarima_dict[key],
        }
        df = df.append(new_row, ignore_index=True)
    df.set_index(var.DATE)
    df.index.freq = "MS"
    df.to_csv("../vdata/graph_" + country + "_" + energy + ".csv")


# df = dr.organize_table("France")
# (out, z) = progressive_prediction(df, "demand", "SARIMA")
# print(z)

# col_names = [var.DATE, "predictions"]

# df = pd.DataFrame(columns=col_names)
# df.set_index(var.DATE, inplace=True)
# z = df.loc["x"]
# print(z)

d = {"a": 1, "b": 2, "c": 3}
e = {"a": 1, "b": 2, "c": 3}
df = pd.DataFrame(columns=["key", "d", "e"])
for key in d:
    new_row = {"key": key, "d": d[key], "e": e[key]}
    df = df.append(new_row, ignore_index=True)
df.set_index("key", inplace=True)
print(df)
