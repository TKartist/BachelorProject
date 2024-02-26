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

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    if item_type == 0:
        for item in df["item"].unique():
            item_data = df[df["item"] == item]
            axs[0].plot(item_data.index, item_data["value"], marker="o", label=item)
            axs[1].scatter(item_data.index, item_data["value"], marker="o", label=item)
    elif item_type <= len(energy_array):
        item_data = df[df["item"] == energy_array[item_type - 1]]
        axs[0].plot(
            item_data.index,
            item_data["value"],
            marker="o",
            label=energy_array[item_type - 1],
        )
        axs[1].scatter(
            item_data.index,
            item_data["value"],
            marker="o",
            label=energy_array[item_type - 1],
        )
    else:
        return

    axs[0].set_title("Values by Item Line Plot")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Values")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[1].set_title("Values by Item Scatter Plot")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Values")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)

    plt.show()


visualize_country("Switzerland")
