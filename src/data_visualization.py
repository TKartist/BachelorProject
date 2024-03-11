import data_reader as dr
import pandas as pd
import matplotlib.pyplot as plt


def visualize_country(country_name):
    df = dr.organize_table(country_name)
    print(df.columns)
    col_index = int(
        input(
            "Select the index of the energy source you want to view. \n index starts from 1, choose 0 if want to view all"
        )
    )
    # _, axs = plt.subplots(2, 1, figsize=(16, 10))
    if col_index == 0:
        df.plot(figsize=(16, 10))
    else:
        df[df.columns[col_index - 1]].plot(figsize=(16, 10))
    plt.show()


visualize_country("Switzerland")
