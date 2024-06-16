import pandas as pd
import numpy as np
import os
from variables import DATE, data_dir, rebasement
import matplotlib.pyplot as plt


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


def read_country(country):
    country = clean_filename(country)
    try:
        df = pd.read_csv(data_dir + country)
    except Exception as e:
        print("Please enter a correct filename")
    df[DATE] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    return df


def organize_table(country):
    df = read_country(country)
    new_df = pd.DataFrame()
    new_df[DATE] = df[DATE].unique()
    new_df = new_df.set_index(DATE)

    energy_types = df["item"].unique()
    for energy in energy_types:
        temp_df = df[df["item"] == energy]
        # temporary patch for Greece as it has 2 demand values (DXT and ADMIE -> providers)
        # for now I am only taking the DXT values because it is fully provided
        if energy == "demand" and extract_country_name(country) == "Greece":
            temp_df = temp_df.drop(temp_df[temp_df["provider"] == "ADMIE"].index)
        temp_df = temp_df.set_index(DATE)
        new_df[energy] = temp_df["value"]
        new_df[energy] = new_df[energy].fillna(0)
    new_df.fillna(0, inplace=True)
    return new_df


def organize_rebasement(country, energy, date):
    df = read_country(country)
    temp_df = df[df["item"] == energy].dropna()
    new_df = pd.DataFrame()
    new_df[DATE] = temp_df[DATE].unique()
    new_df = new_df.set_index(DATE)
    temp_df = temp_df.set_index(DATE)
    new_df[rebasement] = temp_df["value"] / temp_df[rebasement]
    new_df = new_df.dropna()
    return new_df[new_df.index <= date]


def organize_volatility(country, energy):
    data = organize_table(country)
    df = data[[energy]].copy()
    window = 3  # Rolling window size in months
    df["Rolling_Std"] = df[energy].rolling(window).std()
    df["Rolling_Mean"] = df[energy].rolling(window).mean()
    df["CV"] = df["Rolling_Std"] / df["Rolling_Mean"]
    return df
