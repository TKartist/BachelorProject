import pandas as pd
import matplotlib.pyplot as plt
import variables as var
from forecasting import sarima_prediction, arima_prediction
from os import listdir
from os.path import isfile, join
from data_reader import organize_table
import numpy as np


def visualize_country(df, col_index, country_name):
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


# visualize_country(df, 5, "Croatia")


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
    series.plot.bar(legend=True, figsize=(16, 10))
    plt.title(country + " " + energy + " forecasting performance")
    plt.xlabel(var.DATE)
    plt.ylabel(var.RMSE)
    plt.show()


def visualize_model_performance_all():
    files = [f for f in listdir(var.result_dir) if isfile(join(var.result_dir, f))]
    df = pd.DataFrame()
    for file in files:
        df1 = pd.read_csv(var.result_dir + file)
        df1.drop(df1[df1[var.MEAN] == 0].index, inplace=True)
        df = pd.concat([df, df1])
    df.drop([var.DATE], axis=1, inplace=True)
    print(df)
    df[var.ARIMA] = df[var.ARIMA] / df[var.MEAN] * 100
    df[var.SARIMA] = df[var.SARIMA] / df[var.MEAN] * 100
    df.boxplot(column=[var.ARIMA, var.SARIMA])
    plt.title("Overall RMSE / MEAN * 100 ARIMA vs SARIMA")
    plt.xlabel("model")
    plt.ylabel("data")
    plt.show()


def madTings():
    # Generate some sample data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the line graph
    plt.plot(x, y, color="blue", label="Sine Wave")

    # Shade the area under the curve
    plt.fill_between(x, y, color="skyblue", alpha=0.3)

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Sine Wave with Shaded Area")

    # Add legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


madTings()
