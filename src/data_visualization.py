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
    series.boxplot(column=[var.SARIMA, var.ARIMA, var.DL, var.SARIMAX])
    plt.xlabel("model")
    plt.ylabel(var.RMSPE)
    plt.show()


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
    plt.title("RMSPE")
    plt.xlabel("model")
    plt.ylabel("data")
    plt.show()


def median_calc(list):
    return (max(list) + min(list)) / 2


def mean_calc(list):
    return sum(list) / len(list)


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
    df["dl_prediction"] = df["dl_prediction"].apply(lambda x: ast.literal_eval(x))
    df["sarimax_prediction"] = df["sarimax_prediction"].apply(
        lambda x: ast.literal_eval(x)
    )

    df["s_mean"] = df["sarima_prediction"].apply(median_calc)
    df["a_mean"] = df["arima_prediction"].apply(median_calc)
    df["d_mean"] = df["dl_prediction"].apply(median_calc)
    df["sx_mean"] = df["sarimax_prediction"].apply(median_calc)

    df["merged"] = df.apply(
        lambda row: row["arima_prediction"]
        + row["sarima_prediction"]
        + row["dl_prediction"]
        + row["sarimax_prediction"],
        axis=1,
    )
    df["min_range"] = df["merged"].apply(min)
    df["max_range"] = df["merged"].apply(max)
    df["mean"] = df["merged"].apply(mean_calc)
    plt.plot(
        df.index, df["s_mean"], color="green", linestyle="--", label="sarima median"
    )
    plt.plot(df.index, df["a_mean"], color="red", linestyle="--", label="arima median")
    plt.plot(df.index, df["d_mean"], color="black", linestyle="--", label="dl median")
    plt.plot(
        df.index, df["sx_mean"], color="yellow", linestyle="--", label="sarimax median"
    )

    plt.fill_between(
        df.index,
        df["max_range"],
        df["min_range"],
        alpha=0.6,
        label="range(predictions)",
    )
    plt.plot(
        df.index,
        df["mean"],
        color="purple",
        linestyle="--",
        label="mean(predictions)",
    )
    plt.plot(
        source.index[72:],
        source[72:],
        color="blue",
        linewidth=1.0,
        label="test data",
    )
    plt.legend(fontsize=12.5, loc="upper left")
    # plt.title(country + " " + energy + " energy production forecast", fontsize=14)
    plt.show()


def visual_narrative(c, e):
    df = organize_table(c)[e]
    view_trend_seasonality(df)
    visualize_pred_margin(c, e)
    visualize_model_performance(c, e)


visual_narrative("France", "solar")
