import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")


# True if stationary, False if non-stationary
def adf_test(series, title=""):
    print(f"Augumented Dickey Fuller test: {title}\n")

    result = adfuller(series.dropna(), autolag="AIC")
    labels = ["ADF test statistics", "p-val", "# lag used", "# observations"]

    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f"critical value ({key})"] = val

    print(out.to_string())

    if result[1] < 0.05:
        print("\nStrong evidence against Null Hypothesis")
        print("Rejecting the Null Hypothesis")
        print("Data has no unit root and is stationary")
        return True
    else:
        print("\nWeak evidence against Null Hypothesis")
        print("Fail to reject the Null Hypothesis")
        print("Data has a unit root and is non-stationary")
        return False
