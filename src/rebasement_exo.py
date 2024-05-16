import pandas as pd
import numpy as np
import variables as var
import ast

df = pd.read_csv(
    "../demand_forecasts/graph_Hungary_demand.csv", index_col="date", parse_dates=True
)
df.index.freq = "MS"

dictionary = {}

df[var.SARIMA] = df[var.SARIMA].apply(lambda x: ast.literal_eval(x))
model = var.SARIMA

for m in df.index:
    for i in range(var.predictionCount):
        key = m + pd.DateOffset(months=i)
        if key in dictionary:
            dictionary[key].append(df[model][m][i])
        else:
            dictionary[key] = [df[model][m][i]]

new_dict = {"date": [], "SARIMA": []}
for key in dictionary:
    new_dict["date"].append(key)
    new_dict["SARIMA"].append(dictionary[key])

df1 = pd.DataFrame(new_dict)
print(df1)
