import pandas as pd
import os

print(pd.__version__)
arr = os.listdir("../data")

for file in arr:
    values = pd.read_csv("../data/" + file)
    print(values)

# values = pd.read_csv("../data/data_Austria.csv")

# print(values)
