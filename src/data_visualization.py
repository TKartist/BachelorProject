import data_reader as dr
import pandas as pd
import matplotlib.pyplot as plt
import variables as var
from os import listdir
from os.path import isfile, join


def visualize_country(df, col_index, country_name):
    # _, axs = plt.subplots(2, 1, figsize=(16, 10))
    if col_index == 0:
        df.plot(
            figsize=(16, 10), title="Energy consumption monthly data " + country_name
        )
    else:
        df[df.columns[col_index - 1]].plot(
            figsize=(16, 10),
            title=country_name
            + " '"
            + df.columns[col_index - 1]
            + "' energy consumption monthly data",
        )
    plt.show()


def visualize_error(series, type, country, energy):
    series[[type, "Mean"]].plot.bar(legend=True, figsize=(12, 8))
    plt.xlabel("date")
    plt.ylabel(type + " against mean")
    plt.title(country + " " + energy + " forecasting RMSE vs Data Mean")
    plt.show()


def visualize_prediction(prediction, data, title):
    prediction.plot(legend=True, figsize=(16, 10))
    data.plot(legend=True)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.show()

def visualize_model_performance(country, energy):
    series = pd.read_csv(var.result_dir + "prediction_" + country + "_" + energy + "_all", index_col="period")
    series.plot.bar(legend=True, figsize=(16, 10))
    plt.title(country + " " + energy + " forecasting performance")
    plt.xlabel("Date")
    plt.ylabel("RMSE")
    plt.show()

# visualize_model_performance("Switzerland", "ror")

def visualize_model_performance_all():
    files = [f for f in listdir(var.result_dir) if isfile(join(var.result_dir, f))]
    df = pd.DataFrame()
    for file in files:
        df1 = pd.read_csv(var.result_dir + file)
        df = pd.concat([df, df1])
    df.drop([var.PERIOD], axis=1, inplace=True)
    df.boxplot(column=[var.ARIMA, var.SARIMA])
    plt.xlabel("model")
    plt.ylabel("data")
    plt.show()

visualize_model_performance_all()

