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

    _, axs = plt.subplots(2, 1, figsize=(16, 10))
    if item_type == 0:
        for item in df["item"].unique():
            item_data = df[df["item"] == item]
            axs[0].plot(
                item_data.year.astype(str) + "." + item_data.month.astype(str),
                item_data["value"],
                marker="o",
                label=item,
            )
            axs[1].scatter(
                item_data.year.astype(str) + "." + item_data.month.astype(str),
                item_data["value"],
                marker="o",
                label=item,
            )
    elif item_type <= len(energy_array):
        item_data = df[df["item"] == energy_array[item_type - 1]]
        axs[0].plot(
            item_data.year.astype(str) + "." + item_data.month.astype(str),
            item_data["value"],
            marker="o",
            label=energy_array[item_type - 1],
        )
        axs[1].scatter(
            item_data.year.astype(str) + "." + item_data.month.astype(str),
            item_data["value"],
            marker="o",
            label=energy_array[item_type - 1],
        )
    else:
        print("Out of index")
        return
    ticks = item_data.year.astype(str) + ".1"
    axs[0].set_title("Values by Item Line Plot " + country_name)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Values")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].set_xticks(ticks)
    axs[1].set_title("Values by Item Scatter Plot " + country_name)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Values")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].set_xticks(ticks)
    plt.show()


country_name = input("Enter the country name: ")
visualize_country(country_name=country_name)
