import data_reader as dr
import pandas as pd
import matplotlib.pyplot as plt


def visualize_country(country_name):
    df = dr.get_country_data(country_name)
    for item in df["item"].unique():
        item_data = df[df["item"] == item]
        plt.scatter(item_data.index, item_data["value"], label=item)
    plt.title("Rebasement Values by Item")
    plt.xlabel("Index")
    plt.ylabel("Rebasement")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


visualize_country("Germany")
