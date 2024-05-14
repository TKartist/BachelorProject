from data_reader import organize_table, organize_rebasement
from forecasting import progressive_prediction
from variables import SARIMAX
import pandas as pd

countries = ["Netherlands", "Romania", "Hungary"]
energy = "demand"

for country in countries:
    df = organize_table(country)
    (result, prediction) = progressive_prediction(df, energy, country, SARIMAX)
    result.to_csv(
        "../demand_forecasts/" + country + "_analysis.csv", index=True, encoding="utf-8"
    )
    df = pd.DataFrame(list(prediction.items()), columns=["date", SARIMAX])
    df = df.set_index("date")
    print(df)
    df.to_csv(
        "../demand_forecasts/" + country + "_predicted_monthly_demand.csv",
        index=True,
        encoding="utf-8",
    )
