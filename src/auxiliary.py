import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
from sklearn.metrics import mean_absolute_percentage_error
import itertools
from statsmodels.tsa.arima.model import ARIMA
import variables as var
from arch import arch_model

warnings.filterwarnings("ignore")


# True if stationary, False if non-stationary
def adf_test(series):

    result = adfuller(series.dropna(), autolag="AIC")
    labels = ["ADF test statistics", "p-val", "# lag used", "# observations"]

    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f"critical value ({key})"] = val

    print(out.to_string())
    return result[1] < 0.05


def rmspe_calculation(data, predictions):
    if len(data) != len(predictions):
        raise ValueError("the length of test set and forecast doesn't match")
    rmspe = np.sqrt(np.mean(((data - predictions) / data) ** 2)) * 100
    return rmspe


# index = range(8, 12)
# predicted_data = [i + 0.1 for i in index]
# actual_data = list(index)

# # Create the DataFrame
# df = pd.DataFrame({"Index": index, "Predicted": predicted_data, "Actual": actual_data})

# # Set the 'Index' column as the DataFrame index
# df.set_index("Index", inplace=True)


def performance_analysis(data, predictions):
    MAPE = mean_absolute_percentage_error(data, predictions)
    RMSPE = rmspe_calculation(data, predictions)
    data_mean = data.mean()
    return {
        var.DATE: data.index[0],
        var.MAPE: MAPE,
        var.RMSPE: RMSPE,
        var.MEAN: data_mean,
    }


# print(performance_analysis(df["Predicted"], df["Actual"]))


# there are obscure cases of auto_arima where the minimum AIC is present after an inverted bell curve
# hence, in this case, we perform a grid search just in case.
# This function will be alter the order, if it wasn't intercepted by auto_arima function as it should
# i.e. (p, d, q) == (0,0,0)
def grid_search(series, order):
    if order[0] > 0 or order[2] > 0:
        model = ARIMA(series, order=order)
        return model
    p = range(1, 5)
    q = range(1, 5)
    best_aic = np.inf
    best_model = None
    pdq = list(itertools.product(p, order[1], q))
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


def evaluate_garch_models(residuals, p_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p, q in itertools.product(p_values, q_values):
        try:
            model = arch_model(residuals, vol="Garch", p=p, q=q)
            results = model.fit(disp="off")
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, q)
                best_model = results
        except:
            continue
    return best_order, best_model
