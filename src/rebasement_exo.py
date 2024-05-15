from data_reader import organize_table, organize_rebasement
import numpy as np
import pandas as pd
from forecasting import sarimax_prediction, sarima_prediction, arima_prediction
import matplotlib.pyplot as plt
import ast
from auxiliary import rmspe_calculation

data = organize_table("Hungary")
re = organize_rebasement("Hungary", "demand", max(data.index))
df = pd.read_csv(
    "../demand_forecasts/Hungary_predicted_monthly_demand.csv",
    index_col="date",
    parse_dates=True,
)

df["SARIMAX"] = df["SARIMAX"].apply(lambda x: ast.literal_eval(x))
l = []
for iter in range(3):
    s = df.index[iter]
    e = s + pd.DateOffset(months=len(df.index) - 1)
    dates = pd.date_range(start=s, end=e, freq="MS")
    z = pd.DataFrame(dates, columns=["date"])
    y = []
    for a in df["SARIMAX"]:
        y.append(a[iter])
    z["predictions"] = y
    z = z.set_index("date")
    z["demand"] = data["demand"][data.index[data.index >= min(z.index)]]
    l.append(rmspe_calculation(z["demand"], z["predictions"]))
    z.index.freq = "MS"
    plt.plot(dates, y, label="M+" + str(iter + 1) + " forecasts", linestyle="--")
print(l)
# for i in df.index:
#     dates = pd.date_range(
#         start=i, end=i + pd.DateOffset(months=(len(df["SARIMAX"][i]) - 1)), freq="MS"
#     )
#     plt.plot(
#         dates, df["SARIMAX"][i], label="predictions from " + str(i), linestyle="--"
#     )
# def median_value(list):
#     return (max(list) + min(list)) / 2


# df["median"] = df["SARIMAX"].apply(median_value)
# print(df.index[0] + pd.DateOffset(months=1))

data["demand"][84:].rename("monthly data").plot(legend=True)
re["rebasement"][84:].rename("hourly data cumulation").plot(legend=True)
# df["median"].rename("sarimax model forecast").plot(legend=True)
plt.legend()
plt.title("SARIMAX model forecast of demand in Hungary")
plt.show()

plt.bar(
    ["M+1 RMSPE", "M+2 RMSPE", "M+3 RMSPE"],
    l,
    width=0.3,
)
plt.show()
