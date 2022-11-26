import streamlit as st
import math
import pandas_datareader as web
import pandas_market_calendars as mcal
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
from keras.models import Sequential
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
#DATABASE
import deta
from deta import Deta

def insert_day(reference_day, predictions, stock):
    """Returns the report on a successful creation, otherwise raises an error"""
    return db.put({"key":reference_day, "predictions":predictions, "stock":stock})

def fetch_all_days():
    "Returns a dict of all periods"
    res=db.fetch()
    return res.items

def get_day(reference_day):
    """If not found, the function will return None"""
    return db.get(reference_day)

DETA_KEY="a0xjvcbe_oG7UFm166tu6a4UvnSbc4hU7geUatytz"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Stocks Prediction | DAX30"
PAGE_ICON = ":de:"
if __name__ == '__main__':
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title("DAX30 Stocks Prediction Machine Learning App")
    st.text("Welcome to my latest DAX30 stocks prediction Machine Learning App! 🚀")

    st.markdown("## DAX 30 Artificial Neural Network Long Short Term Memory (LSTM) Algorithm")

    today = datetime.today().strftime('%Y-%m-%d')
    st.write("Today:", today)
    target_day = st.text_input('Insert here the target date (format: yyyy-mm-dd):')
    if len(target_day)!=0:
        # Create a calendar
        xfra = mcal.get_calendar('XFRA')
        # Show available calendars
        # print(mcal.get_calendar_names())
        days = xfra.schedule(start_date=today, end_date=target_day)
        days = [str(t) for t in days.index]
        days = [elem[:10] for elem in days]
        # days
        st.write("Days elapsing between today and the target date:", len(days))


        companies = ['adidas AG', 'BASF SE', 'Fresenius SE & Co. KGaA', 'Hannover Rück SE', 'Deutsche Post AG', 'Linde plc',
                     'MTU Aero Engines AG', 'Deutsche Bank Aktiengesellschaft', 'RWE Aktiengesellschaft',
                     'Fresenius Medical Care AG & Co. KGaA', 'Bayerische Motoren Werke Aktiengesellschaft',
                     'MERCK Kommanditgesellschaft auf Aktien', 'Deutsche Telekom AG', 'Symrise AG', 'Allianz SE',
                     'Bayer Aktiengesellschaft', 'Continental Aktiengesellschaft', 'Covestro AG',
                     'Siemens Aktiengesellschaft', 'Deutsche Börse AG', 'Siemens Healthineers AG',
                     'Infineon Technologies AG', 'HeidelbergCement AG', 'Beiersdorf Aktiengesellschaft', 'Volkswagen AG',
                     'Siemens Energy AG', 'Daimler Truck Holding AG', 'Airbus SE', 'PUMA SE', 'Zalando SE']
        symbols = ['ADS.DE', 'BAS.DE', 'FRE.DE', 'HNR1.DE', 'DPW.DE', 'LIN.DE', 'MTX.DE', 'DBK.DE', 'RWE.DE', 'FME.DE',
                   'BMW.DE', 'MRK.DE', 'DTE.DE', 'SY1.DE', 'ALV.DE', 'BAYN.DE', 'CON.DE', '1COV.DE', 'SIE.DE', 'DB1.DE',
                   'SHL.DE', 'IFX.DE', 'HEI.DE', 'BEI.DE', 'VOW3.DE', 'ENR.DE', 'DTG.DE', 'AIR.DE', 'PUM.DE', 'ZAL.DE']
        df = pd.DataFrame(list(zip(companies, symbols)), columns=['Company Name', 'Symbol'])
        st.dataframe(df)

        st.markdown("### Training and Testing Phase")

        for symbol in symbols:
            df = web.DataReader(symbol, data_source='yahoo', start='2018-01-01', end=today)

            # plt.figure(figsize=(16, 8))
            # plt.title(str(symbol) + ' Close Price History')
            # plt.plot(df['Close'])
            # plt.xlabel('Date')
            # plt.ylabel('Close price USD ($)')
            # plt.figure()

            data = df.filter(['Close'])
            dataset = data.values
            training_data_len = math.ceil(len(dataset) * .8)
            # training_data_len

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            # scaled_data

            # Create the training dataset
            # Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]
            # Split X_train and y_train
            X_train = []
            y_train = []
            for i in range(60, len(train_data)):  # 60 days
                X_train.append(train_data[i - 60:i, 0])
                y_train.append(train_data[i, 0])

                # print(len(X_train))
                # print(len(np.transpose(X_train)))

            X_train, y_train = np.array(X_train), np.array(y_train)

            # Reshape the data (Neural Networks want three dimensional data)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            # X_train.shape

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, batch_size=1, epochs=1)

            # Creating the testing dataset
            # Create a new array containing scaled values from index 1954 to 2014
            test_data = scaled_data[training_data_len - 60:, :]
            # Create the datasets x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i - 60:i, 0])

            # Convert the data to a numpy array
            x_test = np.array(x_test)

            # Reshape from 2D to 3D
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Get the models predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Get the root mean squared error (RMSE) (How accurate the model actually is)
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            st.write("Root Mean Squared Error (RMSE):", rmse)

            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            fig = plt.figure(figsize=(16, 8))
            plt.title(str(symbol) + ' LSTM Stock Model')
            plt.xlabel('Date')
            plt.ylabel('Close Price USD ($)')
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
            plt.show()
            # fig.savefig(str(symbol)+'.png')
            st.pyplot(fig)

        st.markdown("### Prediction Phase")
        target_days = days  # predictions for these specific days
        for symbol in symbols:
            previsions = []  # predic price, predic price1, ...
            reference_days = []  # days in which predictions are done
            for i in range(0, len(days)):
                symbol_quote = web.DataReader(symbol, data_source='yahoo', start='2018-01-01', end=today)
                new_df = symbol_quote.filter(['Close'])
                last_60_days = new_df[-(60 - i):]
                if i != 0:
                    for j in range(i, 0, -1):
                        last_60_days.loc[len(last_60_days)] = previsions[len(previsions) - j]
                last_60_days_scaled = scaler.transform(last_60_days.values)
                X_test = []
                X_test.append(last_60_days_scaled)
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                pred_price = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price)
                # print(pred_price[0][0])
                previsions.append(pred_price[0][0])
                # target_days.append(time.strftime('%Y-%m-%d', time.localtime(time.time() + (i+1)*24*3600)))
                reference_days.append(today)
            data = pd.DataFrame({'previsions': previsions, 'target_days': target_days, 'reference_days': reference_days})
            data.to_csv(str(symbol) + ' ' + str(today) + '.csv')
            deta = Deta(DETA_KEY)
            db = deta.Base(symbol + "_DAX_30_daily_reports")
            reference_day=today
            predictions_keys = target_days
            predictions_values = previsions
            predictions = {predictions_keys[i]: str(predictions_values[i]) for i in range(len(predictions_keys))} # using dictionary comprehension to convert lists to dictionary

            insert_day(reference_day, predictions, symbol)
            fetch_all_days()  # in case of modifications
            desired_ref_day = get_day(today)
            df = pd.DataFrame.from_dict(desired_ref_day)
            df = df.reset_index()
            df.rename(columns={"key": "reference_days", "index": "target_days"})
            st.dataframe(df)
            # st.write("Resultant dictionary is : " + str(predictions)) # Printing resultant dictionary


