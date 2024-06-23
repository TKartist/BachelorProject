import pandas as pd
import numpy as np
import variables as var
import ast
from forecasting import progressive_prediction, arima_garch
from data_reader import organize_table, organize_rebasement
from data_visualization import visualize_country
import matplotlib.pyplot as plt
from auxiliary import performance_analysis

# df2 = organize_rebasement("Switzerland", "ror", max(df.index))
# print(df2)

# visualize_country(df, 0, "Switzerland")
# df2.plot()

# df = pd.read_csv(
#     "../demand_forecasts/graph_Hungary_demand.csv", index_col="date", parse_dates=True
# )
# df.index.freq = "MS"

# dictionary = {}

# df[var.SARIMA] = df[var.SARIMA].apply(lambda x: ast.literal_eval(x))
# model = var.SARIMA

# for m in df.index:
#     for i in range(var.predictionCount):
#         key = m + pd.DateOffset(months=i)
#         if key in dictionary:
#             dictionary[key].append(df[model][m][i])
#         else:
#             dictionary[key] = [df[model][m][i]]

# new_dict = {"date": [], "SARIMA": []}
# for key in dictionary:
#     new_dict["date"].append(key)
#     new_dict["SARIMA"].append(dictionary[key])

# df1 = pd.DataFrame(new_dict)
# print(df1)
# x = organize_table("Austria")
# z = progressive_prediction(x, "solar", "Austria", "SARIMAX")
# print(z)

df = organize_table("Belgium")[:-3]
df = df["gen_hydro_exps"]
(x, y, z) = arima_garch(df, 3)
df[50:].plot()
y.plot()
plt.show()
