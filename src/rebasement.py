from data_reader import organize_table, organize_rebasement
import numpy as np
import pandas as pd

data = organize_table("France")
rebasement = organize_rebasement("France", "demand")
print(rebasement)
