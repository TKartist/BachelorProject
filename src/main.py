from os import listdir
from os.path import isfile, join
from forecasting import (
    progressive_prediction,
    generate_csv_all,
    generate_csv_area_chart,
)
from data_visualization import visualize_country
from data_reader import organize_table
import variables as var


def energy_type(df):
    print(df.columns)
    col_index = int(
        input(
            "Select the index of the energy source you want to view. \n index starts from 1, choose 0 if want to view all, \n in case of forecasting, you can only choose one energy type: "
        )
    )
    return col_index


def individual_execution(countries):
    print(
        "############################# Choose a country ############################# \n"
    )
    print(countries)
    country = str(
        input("\n Choose a country from the list above and enter the name here: ")
    )
    print(
        "\n ############################################################################ \n"
    )

    print(
        "############################# Choose an option ############################# \n"
    )
    option = int(
        input(
            " 1. Visualize Country Data \n 2. Forecast the country trend \n Enter the number: "
        )
    )
    print(
        "\n ############################################################################ \n"
    )

    df = organize_table(country)
    col_index = energy_type(df)
    if option == 1:
        visualize_country(df, col_index, country)
    elif option == 2:
        col = df.columns[col_index - 1]
        (result_sarima, pred_sar) = progressive_prediction(df, col, None, var.SARIMA)
        (result_arima, pred_ar) = progressive_prediction(df, col, None, var.ARIMA)
        (result_dl, pred_dl) = progressive_prediction(df, col, None, var.DL)
        (result_sarimax, pred_sarx) = progressive_prediction(
            df, col, country, var.SARIMAX
        )
        generate_csv_all(
            result_sarima, result_arima, result_dl, result_sarimax, country, col
        )
        generate_csv_area_chart(pred_sar, pred_ar, pred_dl, pred_sarx, country, col)
    else:
        raise Exception("Please a valid option between 1 or 2")


def predict_all(countries):
    for country in countries:
        df = organize_table(country)
        for col in df.columns:
            (result_sarimax, pred_sarx) = progressive_prediction(
                df, col, country, var.SARIMAX
            )
            (result_sarima, pred_sar) = progressive_prediction(
                df, col, None, var.SARIMA
            )
            (result_arima, pred_ar) = progressive_prediction(df, col, None, var.ARIMA)
            (result_dl, pred_dl) = progressive_prediction(df, col, None, var.DL)
            generate_csv_all(
                result_sarima, result_arima, result_dl, result_sarimax, country, col
            )
            generate_csv_area_chart(pred_sar, pred_ar, pred_dl, pred_sarx, country, col)


def main():
    countries = [
        f
        for f in listdir("../data/")
        if isfile(join("../data/", f)) and f.endswith(".csv")
    ]
    for i in range(len(countries)):
        countries[i] = countries[i][5:-4]
    running = True
    while running:
        print(
            "############################# Choose an option ############################# \n"
        )
        print(
            "Press 1 to produce forecast error report of all countries and energies, Press 2 to do them individually \n",
            "Press 3 to forecast only France and Germany \n Press 4 to finish",
        )
        option = int(input("\n Choose an option: "))
        print(
            "\n ############################################################################ \n"
        )
        if option == 1:
            predict_all(countries)
        elif option == 2:
            individual_execution(countries)
        elif option == 3:
            predict_all(var.M2)
        elif option == 4:
            running = False
        else:
            print("Please choose an option between 1 or 2")


if __name__ == "__main__":
    main()
