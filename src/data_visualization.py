import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import variables as var
from os import listdir
from os.path import isfile, join
from data_reader import organize_table, organize_rebasement, organize_volatility
from statsmodels.tsa.seasonal import seasonal_decompose
import ast
from auxiliary import rmspe_calculation
import seaborn as sns
import math
from scipy.stats import gaussian_kde


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
        var.result_dir + "prediction_" + country + "_" + energy + "_all.csv",
        index_col=var.DATE,
    )
    series.boxplot(column=models)
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
    df.boxplot(column=[var.ARIMA, var.SARIMA, var.DL, var.SARIMAX])
    plt.title("RMSPE")
    plt.xlabel("model")
    plt.ylabel("data")
    plt.show()


def median_calc(list):
    return (max(list) + min(list)) / 2


def mean_calc(list):
    return sum(list) / len(list)


def pred_struct_alter(df, model):
    dictionary = {}
    df[model] = df[model].apply(lambda x: ast.literal_eval(x))

    for m in df.index:
        for i in range(var.predictionCount):
            key = m + pd.DateOffset(months=i)
            if key in dictionary:
                dictionary[key].append(df[model][m][i])
            else:
                dictionary[key] = [df[model][m][i]]

    new_dict = {var.DATE: [], model: []}
    for key in dictionary:
        new_dict[var.DATE].append(key)
        new_dict[model].append(dictionary[key])

    df1 = pd.DataFrame(new_dict).set_index(var.DATE)
    return df1[model]


def visualize_pred_margin(country, energy):
    source = organize_table(country)[energy]
    df = pd.read_csv(
        var.vdata_dir + "graph_" + country + "_" + energy + ".csv",
        index_col=var.DATE,
        parse_dates=True,
    )
    df1 = pd.DataFrame(columns=[var.DATE]).set_index(var.DATE)
    df1.index.freq = "MS"
    df1[var.SARIMA] = pred_struct_alter(df, var.SARIMA)
    df1[var.ARIMA] = pred_struct_alter(df, var.ARIMA)
    df1[var.DL] = pred_struct_alter(df, var.DL)
    df1[var.SARIMAX] = pred_struct_alter(df, var.SARIMAX)
    df = df1
    df["s_mean"] = df[var.SARIMA].apply(median_calc)
    df["a_mean"] = df[var.ARIMA].apply(median_calc)
    df["d_mean"] = df[var.DL].apply(median_calc)
    df["sx_mean"] = df[var.SARIMAX].apply(median_calc)

    df["merged"] = df.apply(
        lambda row: row[var.ARIMA] + row[var.SARIMA] + row[var.DL] + row[var.SARIMAX],
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
    visualize_model_performance(c, e, var.models)


def plot_heatmaps(results_list, countries_list, title):
    _, axs = plt.subplots(2, 3, figsize=(16, 10))

    for i, (ax, result) in enumerate(zip(axs.flat, results_list)):
        sns.heatmap(
            result,
            cmap="viridis",
            ax=ax,
            cbar=False,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 25},
        )
        ax.set_title(countries_list[i], fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=20)
    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.03, right=0.98, hspace=0.25, wspace=0.1
    )
    plt.savefig(f"../figs/heat_map_{title}.png")
    plt.show()


def visualize_heat_map():
    folder_path = "../results/"
    country_datas = []
    for c in var.collected_countries:
        country_model_mean = {}
        data = organize_table(c)
        energies = data.columns
        for e in energies:
            df = pd.read_csv(folder_path + "prediction_" + c + "_" + e + "_all.csv")
            country_model_mean[e[:3]] = {
                var.ARIMA: math.log2(df[var.ARIMA].mean()),
                var.SARIMA: math.log2(df[var.SARIMA].mean()),
                var.DL: math.log2(df[var.DL].mean()),
                var.SARIMAX: math.log2(df[var.SARIMAX].mean()),
            }
        results = pd.DataFrame(country_model_mean).transpose()
        country_datas.append(results)
    country_results_part1 = country_datas[:6]
    country_results_part2 = country_datas[6:]
    countries_part1 = var.collected_countries[:6]
    countries_part2 = var.collected_countries[6:]
    plot_heatmaps(country_results_part1, countries_part1, "part1")
    plot_heatmaps(country_results_part2, countries_part2, "part2")


def distribution_graph_logged():
    data_a, data_s, data_sx, data_d = [], [], [], []
    folder_path = "../results/"
    for c in var.collected_countries:
        d = organize_table(c)
        energies = d.columns
        for e in energies:
            df = pd.read_csv(folder_path + "prediction_" + c + "_" + e + "_all.csv")
            data_a.append(math.log2(df[var.ARIMA].mean()))
            data_s.append(math.log2(df[var.SARIMA].mean()))
            data_sx.append(math.log2(df[var.SARIMAX].mean()))
            data_d.append(math.log2(df[var.DL].mean()))
    data_a = np.sort(np.array(data_a))
    data_s = np.sort(np.array(data_s))
    data_sx = np.sort(np.array(data_sx))
    data_d = np.sort(np.array(data_d))
    x = len(data_a)
    third_quartile = int(x * 0.75)
    second_quartile = int(x * 0.5)

    _, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].hist(data_a, bins=50, edgecolor="black", alpha=0.2, density=True)
    axes[0, 0].axvline(
        data_a[second_quartile],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="2nd Quartile",
    )
    axes[0, 0].axvline(
        data_a[third_quartile],
        color="green",
        linestyle="--",
        linewidth=2,
        label="3rd Quartile",
    )
    axes[0, 0].set_title("logged ARIMA forecast accuracy density graph", fontsize=20)
    axes[0, 0].legend()

    axes[0, 1].hist(data_s, bins=50, edgecolor="black", alpha=0.2, density=True)
    axes[0, 1].axvline(
        data_s[second_quartile],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="2nd Quartile",
    )
    axes[0, 1].axvline(
        data_s[third_quartile],
        color="green",
        linestyle="--",
        linewidth=2,
        label="3rd Quartile",
    )
    axes[0, 1].set_title("logged SARIMA forecast accuracy density graph", fontsize=20)
    axes[0, 1].legend()

    axes[1, 0].hist(data_sx, bins=50, edgecolor="black", alpha=0.2, density=True)
    axes[1, 0].axvline(
        data_sx[second_quartile],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="2nd Quartile",
    )
    axes[1, 0].axvline(
        data_sx[third_quartile],
        color="green",
        linestyle="--",
        linewidth=2,
        label="3rd Quartile",
    )
    axes[1, 0].set_title("logged SARIMAX forecast accuracy density graph", fontsize=20)
    axes[1, 0].legend()

    axes[1, 1].hist(data_d, bins=50, edgecolor="black", alpha=0.2, density=True)
    axes[1, 1].axvline(
        data_d[second_quartile],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="2nd Quartile",
    )
    axes[1, 1].axvline(
        data_d[third_quartile],
        color="green",
        linestyle="--",
        linewidth=2,
        label="3rd Quartile",
    )
    print(data_sx[second_quartile], data_sx[third_quartile])
    print(data_d[second_quartile], data_d[third_quartile])
    print(data_a[second_quartile], data_a[third_quartile])
    print(data_s[second_quartile], data_s[third_quartile])

    axes[1, 1].set_title("logged DL forecast accuracy density graph", fontsize=20)
    axes[1, 1].legend()

    # Adjust tick parameters
    for ax in axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=20)

    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.03, right=0.98, hspace=0.25, wspace=0.1
    )
    plt.savefig(f"../figs/forecast_accuracy_density.png")
    plt.show()


def visualize_volatility(countries, energies):
    for i in range(len(countries)):
        x = organize_volatility(countries[i], energies[i])
        plt.plot(x["CV"], label=f"volatility {countries[i]} {energies[i]}")
    plt.title("Coefficient Variance (Volatility) Visualization")
    plt.legend()
    plt.show()


visualize_volatility(
    countries=["France", "France", "Netherlands"], energies=["demand", "solar", "hydro"]
)


def iterative_forecast_visualization(energy, country, models):
    data = organize_table(country)[energy]
    exogenous = organize_rebasement(country, energy, max(data.index))
    path = "../vdata/"
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
        plt.show()
        plt.title(model + " M+alpha RMSPE of " + energy + " in " + country)
        plt.bar(
            ["M+1 RMSPE", "M+2 RMSPE", "M+3 RMSPE"],
            rmspes,
            width=0.3,
        )
        plt.show()
    title = ""
    for model in models:
        title = title + model + ", "
    title = title + "performance of " + energy + " forecast in " + country
    plt.title(title)
    series = pd.read_csv(
        var.result_dir + "prediction_" + country + "_" + energy + "_all.csv",
        index_col=var.DATE,
    )
    # series.boxplot(column=[var.SARIMA, var.ARIMA, var.DL, var.SARIMAX])
    series.boxplot(column=models)
    plt.xlabel("model")
    plt.ylabel(var.RMSPE)
    plt.show()


# iterative_forecast_visualization(
#     "gen_wind", "Netherlands", ["ARIMA", "SARIMAX", "SARIMA", "DL"]
# )
# visualize_pred_margin("Hungary", "demand")
# visualize_model_performance_all()
