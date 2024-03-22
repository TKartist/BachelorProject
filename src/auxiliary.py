import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
from statsmodels.tsa.arima.model import ARIMA

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


def performance_analysis(data, predictions):
    MAE = mean_absolute_error(data, predictions)
    MSE = mean_squared_error(data, predictions)
    RMSE = np.sqrt(MSE)
    data_mean = data.mean()
    return (MAE, MSE, RMSE, data_mean)


# there are obscure cases of auto_arima where the minimum AIC is present after an inverted bell curve
# hence, in this case, we perform a grid search just in case.
# This function will be alter the order, if it wasn't intercepted by auto_arima function as it should
# i.e. (p, d, q) == (0,0,0)
def grid_search(series, order):
    if order[0] > 0 or order[2] > 0:
        # perform grid_search
        model = ARIMA(series, order=order)
        return model
    p = range(1, 7)
    q = range(1, 7)
    best_aic = np.inf
    best_model = None
    pdq = list(itertools.product(p, order, q))
    for param in pdq:
        try:
            model = ARIMA(series, order=param)
            results = model.fit()
            aic = results.aic

            if results.mle_retvals["converged"] and aic < best_aic:
                best_model = model
                best_aic = aic
        except:
            continue
    return best_model
