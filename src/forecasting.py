import pandas as pd
import numpy as np
import keras

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from auxiliary import performance_analysis, evaluate_garch_models
from data_reader import organize_rebasement
import variables as var
from variables import predictionCount
import warnings
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from sklearn.preprocessing import MinMaxScaler
from keras_preprocessing.sequence import TimeseriesGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM
from keras.src.callbacks import EarlyStopping

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Non-invertible|Non-stationary"
)


def arima_garch(series, test_size):
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]
    print(test)
    for mod in var.optimizers:
        stepwise = auto_arima(
            train,
            test="adf",
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=1,
            max_d=3,
            seasonal=False,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            method=mod,
            maxiter=1000,
        )

        start = len(train)
        end = len(train) + len(test) - 1

        model = ARIMA(series, order=stepwise.order)
        results = model.fit()

        if results.mle_retvals["converged"]:
            break
    predictions = results.predict(start=start, end=end).rename(stepwise.order)

    return (test, predictions, stepwise.order)


def arima_prediction(series, test_size):
    # train and test set split (assuming we try to predic half a year)
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]
    for mod in var.optimizers:
        stepwise = auto_arima(
            train,
            test="adf",
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            max_d=5,
            seasonal=False,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method=mod,
            maxiter=100,
        )

        start = len(train)
        end = len(train) + len(test) - 1

        model = ARIMA(series, order=stepwise.order)
        results = model.fit()

        if results.mle_retvals["converged"]:
            break
    predictions = results.predict(start=start, end=end, typ="levels").rename(
        stepwise.order
    )
    return (test, predictions, stepwise.order)


# Seasonal AutoRegressive Integrated Moving Average Model
def sarima_prediction(series, test_size):
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]
    for mod in var.optimizers:
        stepwise = auto_arima(
            train,
            m=12,
            seasonal=True,
            test="adf",
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            max_d=5,
            start_P=0,
            start_Q=0,
            start_D=0,
            max_P=2,
            max_Q=2,
            max_D=5,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method=mod,
            maxiter=100,
        )

        order, seasonal_order = stepwise.order, stepwise.seasonal_order

        start = len(train)
        end = len(train) + len(test) - 1
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit()

        if results.mle_retvals["converged"]:
            print("goddammmmm2")
            break
    title = str(order) + "X" + str(seasonal_order)
    predictions = results.predict(start, end, typ="levels").rename(title)
    return (test, predictions, title)


def sarimax_prediction(series, test_size, exogenous):
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]
    exo_train = exogenous[:bound]
    exo_test = exogenous[bound:]
    for mod in var.optimizers:
        stepwise = auto_arima(
            train,
            exogenous=exo_train,
            m=12,
            seasonal=True,
            test="adf",
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            max_d=5,
            start_P=0,
            start_Q=0,
            start_D=0,
            max_P=2,
            max_Q=2,
            max_D=5,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method=mod,
            maxiter=100,
        )
        order, seasonal_order = stepwise.order, stepwise.seasonal_order

        start = len(train)
        end = len(train) + len(test) - 1
        model = SARIMAX(
            train,
            exog=exo_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit()
        if results.mle_retvals["converged"]:
            break
    title = str(order) + "X" + str(seasonal_order)
    predictions = results.predict(start, end, exog=exo_test, typ="levels")
    print(predictions)
    return (test, predictions, title)


class CustomStopper(keras.callbacks.Callback):
    def __init__(self, desired_loss):
        super(CustomStopper, self).__init__()
        self.desired_loss = desired_loss

    def on_epoch_end(self, epoch, logs={}):
        if logs.get("loss") <= self.desired_loss:
            print(f"\nReached desired loss ({self.desired_loss}), stopping training!")
            self.model.stop_training = True


def dl_forecast(series, test_size):
    bound = len(series) - test_size
    train = series[:bound]
    test = series[bound:]

    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    n_input = 24
    if len(scaled_train) < 24:
        n_input = len(scaled_train)

    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input)
    model = Sequential()
    model.add(
        LSTM(250, activation="relu", input_shape=(n_input, 1), return_sequences=True)
    )
    model.add(LSTM(150, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="MSE")
    X, y = generator[0]
    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, min_delta=0.01
    )
    stopper = CustomStopper(0.005)
    model.fit(x=X, y=y, epochs=50, batch_size=1, callbacks=[early_stop, stopper])

    test_preds = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, 1))
    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_preds.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_pred = scaler.inverse_transform(test_preds)
    test["Predictions"] = true_pred
    return test


def progressive_prediction(df, energy, country, pred_algo):
    target = df[energy].drop(df[df[energy] == 0].index)
    numberOfPredictions = 6
    start = int(len(target) - numberOfPredictions)
    pred_col = {}
    out = pd.DataFrame(columns=[var.DATE, var.MAPE, var.RMSPE, var.MEAN, var.order])
    for i in range(start + 1, len(target) + 1):
        if pred_algo == var.SARIMA:
            (test, pred, order) = sarima_prediction(target[:i], predictionCount)
        elif pred_algo == var.ARIMA:
            (test, pred, order) = arima_prediction(target[:i], predictionCount)
        elif pred_algo == var.SARIMAX:
            exo = organize_rebasement(country, energy, max(target[:i].index))
            (test, pred, order) = sarimax_prediction(target[:i], predictionCount, exo)
        else:
            dl_df = pd.DataFrame(target[:i], columns=[energy])
            dl_out = dl_forecast(dl_df, predictionCount)
            pred = dl_out["Predictions"]
            test = dl_out[energy]
            order = None
        performance = performance_analysis(test, pred)
        performance[var.order] = order
        out = out.append(performance, ignore_index=True)
        initial_index = pred.index[0]
        for ind in pred.index:
            if initial_index in pred_col:
                pred_col[initial_index].append(pred[ind])
            else:
                pred_col[initial_index] = [pred[ind]]
    out = out.set_index(var.DATE)
    return (out, pred_col)


def generate_csv(series, country, energy):
    series.to_csv(
        var.result_dir + "prediction_" + country + "_" + energy + ".csv",
        encoding="utf-8",
    )


def generate_csv_all(
    sarima_series, arima_series, dl_series, sarimax_series, country, energy
):
    df = pd.DataFrame(index=arima_series.index)
    df[var.MEAN] = sarima_series[var.MEAN]
    df[var.SOURCE] = country + "_" + energy
    df[var.ARIMA] = arima_series[var.RMSPE]
    df[var.SARIMA] = sarima_series[var.RMSPE]
    df[var.DL] = dl_series[var.RMSPE]
    df[var.SARIMAX] = sarimax_series[var.RMSPE]
    df[var.ARIMAM] = arima_series[var.MAPE]
    df[var.SARIMAM] = sarima_series[var.MAPE]
    df[var.SARIMAXM] = sarimax_series[var.MAPE]
    df[var.DLM] = dl_series[var.MAPE]
    df[var.order + "X" + var.ARIMA] = arima_series[var.order]
    df[var.order + "X" + var.SARIMA] = sarima_series[var.order]
    df[var.order + "X" + var.SARIMAX] = sarimax_series[var.order]
    df.to_csv(
        "../results/prediction_" + country + "_" + energy + "_all.csv",
        encoding="utf-8",
    )


def generate_csv_area_chart(
    sarima_dict, arima_dict, dl_dict, sarimax_dict, country, energy
):
    df = pd.DataFrame(columns=[var.DATE, var.ARIMA, var.SARIMA, var.DL])
    for key in arima_dict:
        new_row = {
            var.DATE: key,
            var.ARIMA: arima_dict[key],
            var.SARIMA: sarima_dict[key],
            var.DL: dl_dict[key],
            var.SARIMAX: sarimax_dict[key],
        }
        df = df.append(new_row, ignore_index=True)
    df = df.set_index(var.DATE)
    df.index.freq = "MS"
    df.to_csv("../vdata/graph_" + country + "_" + energy + ".csv")
