import pandas as pd
import numpy as np
import os
from variables import DATE, data_dir


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
    filenames = os.listdir(data_dir)
    return filenames


def organize_table(filename):
    filename = clean_filename(filename)
    try:
        df = pd.read_csv(data_dir + filename)
    except Exception as e:
        print("Please enter a correct filename")
    df[DATE] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    energy_types = df["item"].unique()
    new_df = pd.DataFrame()
    new_df[DATE] = df[DATE].unique()
    new_df = new_df.set_index(DATE)
    for energy in energy_types:
        temp_df = df[df["item"] == energy]
        # temporary patch for Greece as it has 2 demand values (DXT and ADMIE -> providers)
        # for now I am only taking the DXT values because it is fully provided
        if energy == "demand" and extract_country_name(filename) == "Greece":
            temp_df = temp_df.drop(temp_df[temp_df["provider"] == "ADMIE"].index)
        temp_df = temp_df.set_index(DATE)
        new_df[energy] = temp_df["value"]
    new_df.fillna(0, inplace=True)
    return new_df


z = organize_table("Austria")
