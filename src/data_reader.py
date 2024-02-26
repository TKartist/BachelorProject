import pandas as pd
import os


def clean_filename(filename):
    if filename[0:4] != "data":
        filename = "data_" + filename
    if filename.find(".") != -1:
        filename = filename[0 : filename.find(".")] + ".csv"
    else:
        filename = filename + ".csv"
    return filename


def get_filenames():
    filenames = os.listdir("../data")
    return filenames


def get_country_data(filename):
    original = filename
    try:
        filename = clean_filename(filename)
        df = pd.read_csv("../data/" + filename)
        return df
    except Exception as e:
        print("Please enter a correct filename, there is no data on " + original)


def get_all_data():
    fnames = get_filenames()
    df_arr = []
    for fname in fnames:
        df_arr.append(get_country_data(fname))
    return df_arr
