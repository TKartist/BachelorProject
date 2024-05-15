import pandas as pd
import matplotlib.pyplot as plt
import variables as var
from os import listdir
from os.path import isfile, join
from data_reader import organize_table, organize_rebasement
from statsmodels.tsa.seasonal import seasonal_decompose
import ast
from auxiliary import rmspe_calculation


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


def visualize_model_performance(country, energy, models):
    series = pd.read_csv(
        # var.result_dir + "prediction_" + country + "_" + energy + "_all.csv",
        "../demand_forecasts/prediction_" + country + "_" + energy + "_all.csv",
        index_col=var.DATE,
    )
    # series.boxplot(column=[var.SARIMA, var.ARIMA, var.DL, var.SARIMAX])
    series.boxplot(column=models)
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


# def visual_narrative(c, e):
#     df = organize_table(c)[e]
#     view_trend_seasonality(df)
#     visualize_pred_margin(c, e)
#     visualize_model_performance(c, e)


# visual_narrative("France", "solar")


def iterative_forecast_visualization(energy, country, models):
    data = organize_table(country)[energy]
    exogenous = organize_rebasement(country, energy, max(data.index))
    path = "../demand_forecasts/"
    df = pd.read_csv(
        path + "graph_" + country + "_" + energy + ".csv",
        index_col="date",
        parse_dates=True,
    )
    for model in models:
        df[model] = df[model].apply(lambda x: ast.literal_eval(x))
        rmspes = []
        for iter in range(3):
            s = df.index[iter]
            e = s + pd.DateOffset(months=len(df.index) - 1)
            dates = pd.date_range(start=s, end=e, freq="MS")
            z = pd.DataFrame(dates, columns=["date"])
            y = []
            for a in df[model]:
                y.append(a[iter])
            z["predictions"] = y
            z = z.set_index("date")
            z["demand"] = data[data.index[data.index >= min(z.index)]]
            rmspes.append(rmspe_calculation(z["demand"], z["predictions"]))
            z.index.freq = "MS"
            plt.plot(
                dates, y, label="M+" + str(iter + 1) + " forecasts", linestyle="--"
            )
        data[84:].rename("monthly data").plot(legend=True)
        exogenous["rebasement"][84:].rename("hourly data cumulation").plot(legend=True)
        # df["median"].rename("sarimax model forecast").plot(legend=True)
        plt.legend()
        plt.title(model + " model forecast of " + energy + " in " + country)
        plt.gcf().set_size_inches(10, 6)
        plt.savefig(
            path + model + " model forecast of " + energy + " in " + country + ".png"
        )
        plt.show()
        plt.title(model + " M+alpha RMSPE of " + energy + " in " + country)
        plt.bar(
            ["M+1 RMSPE", "M+2 RMSPE", "M+3 RMSPE"],
            rmspes,
            width=0.3,
        )
        plt.savefig(
            path + model + " M+alpha RMSPE of " + energy + " in " + country + ".png"
        )
        plt.show()
    title = ""
    for model in models:
        title = title + model + ", "
    title = title + "performance of " + energy + " forecast in " + country
    plt.title(title)
    plt.savefig(path + title + ".png")
    visualize_model_performance(country, energy, models)


iterative_forecast_visualization("demand", "Romania", ["SARIMAX", "SARIMA", "DL"])
