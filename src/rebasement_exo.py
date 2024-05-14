from data_reader import organize_table, organize_rebasement
import numpy as np
import pandas as pd
from forecasting import sarimax_prediction, sarima_prediction, arima_prediction
import matplotlib.pyplot as plt
import ast

data = organize_table("Hungary")
re = organize_rebasement("Hungary", "demand", max(data.index))
df = pd.read_csv(
    "../demand_forecasts/Hungary_predicted_monthly_demand.csv",
    index_col="date",
    parse_dates=True,
)
df["SARIMAX"] = df["SARIMAX"].apply(lambda x: ast.literal_eval(x))


def median_value(list):
    return (max(list) + min(list)) / 2


df["median"] = df["SARIMAX"].apply(median_value)

data["demand"][72:].rename("monthly data").plot(legend=True)
re["rebasement"][72:].rename("hourly data cumulation").plot(legend=True)
df["median"].rename("sarimax model forecast").plot(legend=True)
plt.show()
