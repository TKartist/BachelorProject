import data_reader as dr
import pandas as pd
import matplotlib.pyplot as plt


def visualize_country(country_name):
    df = dr.organize_table(country_name)
    print(df.columns)
    col_index = int(
        input(
            "Select the index of the energy source you want to view. \n index starts from 1, choose 0 if want to view all: "
        )
    )
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


def visualize_error(series, type):
    series[[type, "Mean"]].plot.bar(legend=True, figsize=(12, 8))
    plt.xlabel("date")
    plt.ylabel(type + " against mean")
    plt.show()
