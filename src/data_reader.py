import pandas as pd
import numpy as np
import os


def clean_filename(filename):
    if filename[0:4] != "data":
        filename = "data_" + filename
    if filename.find(".") != -1:
        filename = filename[0 : filename.find(".")] + ".csv"
    else:
        filename = filename + ".csv"
    return filename


def extract_country_name(filename):
    return filename[filename.find("_") + 1 : filename.find(".")]


def get_filenames():
    filenames = os.listdir("../data")
    return filenames


def organize_table(filename):
    filename = clean_filename(filename)
    try:
        df = pd.read_csv("../data/" + filename)
    except Exception as e:
        print("Please enter a correct filename")
    df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    energy_types = df["item"].unique()
    new_df = pd.DataFrame()
    new_df["date"] = df["date"].unique()
    new_df = new_df.set_index("date")
    for energy in energy_types:
        temp_df = df[df["item"] == energy]
        temp_df = temp_df.set_index("date")
        new_df[energy] = temp_df["value"]
    return new_df


df = organize_table("Switzerland")
print(df.columns[0])
