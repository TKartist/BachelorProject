import data_reader as dr
import pandas as pd
import matplotlib.pyplot as plt


def visualize_country(country_name):
    df = dr.get_country_data(country_name)
    energy_array = df["item"].unique()
    print(energy_array)
    item_type = int(
        input(
            "Which of the energy sources would you like to visualize, \n please enter 0 if all, otherwise input index of the item (from 1): "
        )
    )

    if item_type == 0:
        for item in df["item"].unique():
            item_data = df[df["item"] == item]
            plt.scatter(item_data.index, item_data["value"], label=item)
    elif item_type <= len(energy_array):
        item_data = df[df["item"] == energy_array[item_type - 1]]
        plt.scatter(
            item_data.index, item_data["value"], label=energy_array[item_type - 1]
        )
    else:
        return
    plt.title("Values by Item")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


visualize_country("Switzerland")
