import pandas as pd
import matplotlib.pyplot as plt
import variables as var
from os import listdir
from os.path import isfile, join
from data_reader import organize_table
from statsmodels.tsa.seasonal import seasonal_decompose
import ast


def view_trend_seasonality(series):
    results = seasonal_decompose(series)
    results.plot()
    plt.show()


df = organize_table("France")["demand"]
view_trend_seasonality(df)


def visualize_country(df, col_index, country_name):
    if col_index == 0:
        df.plot(
            figsize=(16, 10),
            title="Energy consumption monthly data " + country_name,
            marker="o",
        )
    else:
        df[df.columns[col_index - 1]].plot(
            figsize=(16, 10),
            title=country_name
            + " '"
            + df.columns[col_index - 1]
            + "' energy consumption monthly data",
            marker="o",
        )
    plt.show()


def visualize_error(series, type, country, energy):
    series[[type, var.MEAN]].plot.bar(legend=True, figsize=(12, 8))
    plt.xlabel(var.DATE)
    plt.ylabel(type + " against mean")
    plt.title(country + " " + energy + " forecasting RMSE vs Data Mean")
    plt.show()


def visualize_prediction(prediction, data, title):
    data.plot(legend=True, figsize=(16, 10))
    prediction.plot(legend=True)
    plt.title(title)
    plt.xlabel(var.DATE)
    plt.ylabel("Consumption")
    plt.show()


def visualize_model_performance(country, energy):
    series = pd.read_csv(
        var.result_dir + "prediction_" + country + "_" + energy + "_all.csv",
        index_col=var.DATE,
    )
    series.boxplot(column=[var.SARIMA, var.ARIMA])
    plt.title(country + " " + energy + " forecasting performance")
    plt.xlabel("model")
    plt.ylabel(var.RMSPE)
    plt.show()


# visualize_model_performance("France", "demand")


def visualize_model_performance_all():
    files = [f for f in listdir(var.result_dir) if isfile(join(var.result_dir, f))]
    df = pd.DataFrame()
    for file in files:
        df1 = pd.read_csv(var.result_dir + file)[2:]
        df1.drop(df1[df1[var.MEAN] == 0].index, inplace=True)
        df = pd.concat([df, df1])
    df.drop([var.DATE], axis=1, inplace=True)
    df[var.ARIMA] = df[var.ARIMA] / df[var.MEAN] * 100
    df[var.SARIMA] = df[var.SARIMA] / df[var.MEAN] * 100
    df.boxplot(column=[var.ARIMA, var.SARIMA])
    plt.title("Overall RMSE / MEAN * 100 ARIMA vs SARIMA")
    plt.xlabel("model")
    plt.ylabel("data")
    plt.show()


def firstIteration(list):
    return (max(list) + min(list)) / 2


def visualize_pred_margin(country, energy):
    source = organize_table(country)[energy]
    df = pd.read_csv(
        var.vdata_dir + "graph_" + country + "_" + energy + ".csv",
        index_col=var.DATE,
        parse_dates=True,
    )
    df["sarima_prediction"] = df["sarima_prediction"].apply(
        lambda x: ast.literal_eval(x)
    )
    df["arima_prediction"] = df["arima_prediction"].apply(lambda x: ast.literal_eval(x))
    df["s_mean"] = df["sarima_prediction"].apply(firstIteration)
    df["a_mean"] = df["arima_prediction"].apply(firstIteration)
    df["merged"] = df.apply(
        lambda row: row["arima_prediction"] + row["sarima_prediction"], axis=1
    )
    df["min_range"] = df["merged"].apply(min)
    df["max_range"] = df["merged"].apply(max)
    print(df["min_range"])
    print(df["max_range"])
    plt.plot(df.index, df["s_mean"], color="green", label="sarima median")
    plt.plot(df.index, df["a_mean"], color="red", label="arima median")
    plt.fill_between(
        df.index, df["max_range"], df["min_range"], alpha=0.6, label="prediction region"
    )
    plt.plot(
        source.index[80:], source[80:], color="blue", linewidth=1.0, label="test data"
    )
    plt.legend()
    plt.title(country + " " + energy + " energy production forecast")
    plt.show()


# visualize_pred_margin("France", "wind")
