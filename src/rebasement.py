from data_reader import organize_table, organize_rebasement
import numpy as np
import pandas as pd
from forecasting import sarimax_prediction, sarima_prediction, arima_prediction
import matplotlib.pyplot as plt

data = organize_table("France")
df = organize_rebasement("France", "demand", max(data.index))
# print(df)
prediction = sarimax_prediction(data["demand"], 6, df["rebasement"]).rename("SARIMAX")
(_, z, _) = sarima_prediction(data["demand"], 6)
(_, l, _) = arima_prediction(data["demand"], 6)
# data["hydro_nops"][60:].plot(legend=True, figsize=(16, 10))
df[80:].plot(legend=True)
prediction.plot(legend=True)
z.plot(legend=True)
l.plot(legend=True)
# print(prediction)
plt.show()
