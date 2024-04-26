import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_reader as dr

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from auxiliary import adf_test, performance_analysis, grid_search
import variables as var
from variables import predictionCount
import warnings
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings(
    "ignore", category=UserWarning, message="Non-invertible|Non-stationary"
)


def arima_prediction(series, test_size):
    arima = auto_arima(
        series,
        test="adf",
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        max_d=5,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
    )
    # train and test set split (assuming we try to predic half a year)

    train = series[:-test_size]
    test = series[-test_size:]

    start = len(train)
    end = len(train) + len(test)

    model = ARIMA(series, order=arima.order)
    results = model.fit()

    predictions = results.predict(start=(start + 1), end=end, typ="levels").rename(
        arima.order
    )
    return (test, predictions, arima.order)


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
        max_d=5,
        start_P=0,
        start_Q=0,
        max_P=5,
        max_Q=2,
        max_D=5,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
    )
    order = sarima.order
    seasonal_order = sarima.seasonal_order

    train = series[:-test_size]
    test = series[-test_size:]

    start = len(train)
    end = len(train) + len(test)
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()
    title = str(order) + "X" + str(seasonal_order)
    predictions = results.predict(start + 1, end, typ="levels").rename(title)
    return (test, predictions, title)


def progressive_prediction(df, energy, pred_algo):
    target = df[energy].drop(df[df[energy] == 0].index)
    numberOfPredictions = 6
    start = int(len(target) - numberOfPredictions)
    pred_col = {}
    out = pd.DataFrame(columns=[var.DATE, var.MAPE, var.RMSPE, var.MEAN, var.order])
    for i in range(start, len(target)):
        if pred_algo == var.SARIMA:
            (test, pred, order) = sarima_prediction(target[:i], predictionCount)
        else:
            (test, pred, order) = arima_prediction(target[:i], predictionCount)
        performance = performance_analysis(test, pred)
        performance[var.order] = order
        out = out.append(performance, ignore_index=True)
        print(out)
        for ind in pred.index:
            if ind in pred_col:
                pred_col[ind].append(pred[ind])
            else:
                pred_col[ind] = [pred[ind]]
    out = out.set_index(var.DATE)
    return (out, pred_col)


def generate_csv(series, country, energy):
    series.to_csv(
        var.result_dir + "prediction_" + country + "_" + energy + ".csv",
        encoding="utf-8",
    )


def generate_csv_all(sarima_series, arima_series, country, energy):
    df = pd.DataFrame(index=arima_series.index)
    df[var.MEAN] = sarima_series[var.MEAN]
    df[var.SOURCE] = country + "_" + energy
    df[var.ARIMA] = arima_series[var.RMSPE]
    df[var.SARIMA] = sarima_series[var.RMSPE]
    df[var.ARIMAM] = arima_series[var.MAPE]
    df[var.SARIMAM] = sarima_series[var.MAPE]
    df[var.order + "X" + var.ARIMA] = arima_series[var.order]
    df[var.order + "X" + var.SARIMA] = sarima_series[var.order]
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
    df = df.set_index(var.DATE)
    df.index.freq = "MS"
    df.to_csv("../vdata/graph_" + country + "_" + energy + ".csv")


df = dr.organize_table("France")
# (out, z) = progressive_prediction(df, "demand", "SARIMA")
# print(z)

# col_names = [var.DATE, "predictions"]

# df = pd.DataFrame(columns=col_names)
# df.set_index(var.DATE, inplace=True)
# z = df.loc["x"]
# print(z)
