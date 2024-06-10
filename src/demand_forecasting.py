from data_reader import organize_table, organize_rebasement
from forecasting import progressive_prediction
from variables import SARIMAX
import pandas as pd

# countries = ["Netherlands", "Romania", "Hungary"]
# energy = "demand"

# for country in countries:
#     df = organize_table(country)
#     (result, prediction) = progressive_prediction(df, energy, country, SARIMAX)
#     result.to_csv(
#         "../demand_forecasts/" + country + "_analysis.csv", index=True, encoding="utf-8"
#     )
#     df = pd.DataFrame(list(prediction.items()), columns=["date", SARIMAX])
#     df = df.set_index("date")
#     print(df)
#     df.to_csv(
#         "../demand_forecasts/" + country + "_predicted_monthly_demand.csv",
#         index=True,
#         encoding="utf-8",
#     )

# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# # Example DataFrame
# data = {
#     "A": [1.234, 2.345, 3.456, 4.567, 5.678, 6.789, 7.890, 8.901],
#     "B": [4.567, 5.678, 6.789, 7.890, 8.901, 9.012, 10.123, 11.234],
#     "C": [7.890, 8.901, 9.012, 10.123, 11.234, 12.345, 13.456, 14.567],
#     "D": [10.234, 11.345, 12.456, 13.567, 14.678, 15.789, 16.890, 17.901],
#     "E": [13.678, 14.789, 15.890, 16.901, 17.012, 18.123, 19.234, 20.345],
#     "F": [16.890, 17.901, 18.012, 19.123, 20.234, 21.345, 22.456, 23.567],
#     "G": [20.123, 21.234, 22.345, 23.456, 24.567, 25.678, 26.789, 27.890],
#     "H": [23.456, 24.567, 25.678, 26.789, 27.890, 28.901, 29.012, 30.123],
# }
# df = pd.DataFrame(data)

# # Define sections (slicing the DataFrame into 2x2 grids for the 4x4 plot)
# sections = {
#     f"Section {i}-{j}": df.iloc[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
#     for i in range(4)
#     for j in range(4)
# }

# # Plot sections using subplots
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# plt.rcParams.update({"font.size": 3})

# # Plot each section
# for ax, (title, section) in zip(axes.flatten(), sections.items()):
#     sns.heatmap(
#         section,
#         ax=ax,
#         cmap="viridis",
#         annot=True,
#         annot_kws={"size": 3},
#         xticklabels=True,
#         yticklabels=True,
#     )
#     ax.set_title(title, fontsize=5)  # Setting title font size to 8
#     ax.set_xlabel("X Label", fontsize=5)  # Decrease x-label font size
#     ax.set_ylabel("Y Label", fontsize=5)

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

# df = organize_table("France")["demand"]
# df = df["demand"] / df["demand"].max()
# print(df / df.max())
