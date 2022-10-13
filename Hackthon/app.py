import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler


# --- SET PAGES ---
st.set_page_config(page_title="BTC-Price Predictor", page_icon='🎯', layout="centered")

# ---- Load Model ---
#model = tf.keras.models.load_model("model_lstm_btc.h5")
model = tf.keras.models.load_model('model_lstm_btc.h5')
model_cl_30 = tf.keras.models.load_model('model_lstm_btc_30.h5')
# --- Set Tittle Page ---
st.title('Crypto-Price-Predictor 🎯')
st.markdown('#')

# --- Define Scalar ---
scaled = MinMaxScaler(feature_range=(0,1))

# --- Sidebar ---
st.sidebar.header("")
pilihan = st.sidebar.radio('Please select for ??', ['Visualization', 'Prediction'])

# --- Import Data ---
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df
df = load_data("dataset\second question\Binance_BTCUSDT_d.csv")

df = df.sort_values(by=['date'])

if pilihan == 'Visualization':
    # Create linecahart in plottly
    st.subheader("Whole period of timeframe of Bitcoin close price 2018-2022")
    fig_olhc = px.line(df, x="date", y="close")
    fig_olhc.add_scatter(x=df['date'], y=df['low'], mode='lines',name='Low')
    fig_olhc.add_scatter(x=df['date'], y=df['close'], mode='lines',name='Close')
    fig_olhc.add_scatter(x=df['date'], y=df['high'], mode='lines',name='High')
    fig_olhc.add_scatter(x=df['date'], y=df['open'], mode='lines',name='Open')
    fig_olhc.update_xaxes(title_text='Date')
    fig_olhc.update_yaxes(title_text='BTC Price')

    fig_olhc.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig_olhc)
    st.subheader("Volume periode of timeframe of Bitcoin")
    # Showing volume in BTC
    fig_volume = px.line(df, x=df['date'], y="Volume BTC", title="BTC Volume")
    fig_volume.update_xaxes(title_text='Date')
    fig_volume.update_yaxes(title_text='BTC Price')

    st.plotly_chart(fig_volume)

if pilihan == 'Prediction':
    st.sidebar.header('Setting')
    periods = st.sidebar.select_slider('Select Forecast Time (days)', ['30', '60'])
    model_select = st.sidebar.multiselect('Choice Model', ['LSTM', 'GRU', 'CNN'], default=["LSTM"])
    pair = st.sidebar.multiselect('Select Pair Crypto', ['BTC', 'ETH', 'SOL', 'SHIB'], default=['BTC'])
    
    # --- Logical in periods ---
    if periods == 30:
        x_future = periods
    else:
        x_future = periods 

    # --- Handling Missing Value ---
    df = df.dropna() 

    # ---- Drop fitur ---
    # Convert Date column to datetime form
    df['date'] = pd.to_datetime(df['date'])
    # Set Date column as index
    df = df.set_index(df['date'])
    # Delete the initial Date column because Date has been used as an index, besides that, drop the Open, High, Low, Adj Close, and Volume columns
    # Because it will only use the Close column
    df.drop(columns=['date', 'unix', 'symbol', 'open', 'high', 'low', 'Volume BTC', 'Volume USDT', 'tradecount'], inplace=True)

    # Convert dataframe to numpy array
    dataset = df.values

    # Scaling dataset
    scaled = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaled.fit_transform(dataset)

    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = int(x_future)

    # Split training data into training data sets and train
    # As a first step, get the number of rows to train the model on 80% data
    train_data_length = math.ceil(len(dataset) * 0.8 )

    # Create training and test data
    train_data = data_scaled[0:train_data_length, :]
    test_data = data_scaled[train_data_length - sequence_length:, :]

    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) # Contains sequence_length values 0-sequence_length * columns
            y.append(data[i, 0]) # Contains the prediction values for validation (3rd column = Close),  for single-step prediction

        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y

    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    # Reshape data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # ------------------------- END PREPROCESSING ----------------------------

    next_future = st.radio('Predict Future BTC Prices', ['30 days', '60 days'])
    if next_future == '30 days':
        # Prediction 
        x_future = int(x_future)
        predictions = np.array([])
        last = x_test[-1]
        for i in range(x_future):
            forecast = model_cl_30.predict(np.array([last]))
            last = np.concatenate([last[1:], forecast])
            predictions = np.concatenate([predictions, forecast[0]])
        predictions = scaled.inverse_transform([predictions])[0]
        dicts = []
        forecast_date = df.index[-1]
        for i in range(x_future):
            forecast_date = forecast_date + timedelta(days=1)
            dicts.append({'Predictions':predictions[i], "Date": forecast_date})
        future_lstm = pd.DataFrame(dicts).set_index("Date")
        # Get the predicted values
        y_test_pred_scaled = model_cl_30.predict(x_test)
        # Unscale the predicted values 
        y_test_pred = scaled.inverse_transform(y_test_pred_scaled)
        y_test_unscaled = scaled.inverse_transform(y_test.reshape(-1, 1))
        # Create a graph of predictive result
        train = df[:train_data_length+1]
        test = df[train_data_length:]
        test['Predictions'] = y_test_pred

        st.subheader("Future Prediction BTC Close Prices Next 30 days")
        trace1 = go.Scatter(
            x = train.index,
            y = train['close'],
            mode = 'lines',
            name = 'Data'
        )
        trace2 = go.Scatter(
            x = test.index,
            y = test['Predictions'],
            mode = 'lines',
            name = 'Prediction'
        )
        trace3 = go.Scatter(
            x = test.index,
            y = test['close'],
            mode = 'lines',
            name = 'Ground Truth'
        )
        trace4 = go.Scatter(
            x = future_lstm.index,
            y = future_lstm['Predictions'],
            mode = 'lines',
            name = 'Future Price'
        )
        fig = go.Figure(data=[trace1, trace2, trace3, trace4])
        st.plotly_chart(fig)
    
        st.write("Find out more about the model's performance.")
        if st.button('Click me'):
            st.subheader("Evaluation Model")
            data_eval = pd.DataFrame({
                'Model': ['LSTM'],
                'Eval Loss Train-data': 0.0008,
                'Eval Loss Test-data': 0.0107,
                'MAE Train-data': 0.0107,
                'MAE Test-data': 0.0289,
                'R2-Score Train': 0.99,
                'R2-Score Test-data': 0.94
            })
            st.dataframe(data_eval)
            st.markdown('#')
            st.write("Conclusion from the evaluation results above!")
            st.write("The model we have created is overfitting. The model is quite good at predicting bitcoin price movements for the next 30 days.")

    else:
            # Prediction 
        x_future = int(x_future)
        predictions = np.array([])
        last = x_test[-1]
        for i in range(x_future):
            forecast = model.predict(np.array([last]))
            last = np.concatenate([last[1:], forecast])
            predictions = np.concatenate([predictions, forecast[0]])
        predictions = scaled.inverse_transform([predictions])[0]
        dicts = []
        forecast_date = df.index[-1]
        for i in range(x_future):
            forecast_date = forecast_date + timedelta(days=1)
            dicts.append({'Predictions':predictions[i], "Date": forecast_date})
        future_lstm = pd.DataFrame(dicts).set_index("Date")
        # Get the predicted values
        y_test_pred_scaled = model.predict(x_test)
        # Unscale the predicted values 
        y_test_pred = scaled.inverse_transform(y_test_pred_scaled)
        y_test_unscaled = scaled.inverse_transform(y_test.reshape(-1, 1))
        # Create a graph of predictive result
        train = df[:train_data_length+1]
        test = df[train_data_length:]
        test['Predictions'] = y_test_pred

        st.subheader("Future Prediction BTC Close Prices Next 60 days")
        trace1 = go.Scatter(
            x = train.index,
            y = train['close'],
            mode = 'lines',
            name = 'Data'
        )
        trace2 = go.Scatter(
            x = test.index,
            y = test['Predictions'],
            mode = 'lines',
            name = 'Prediction'
        )
        trace3 = go.Scatter(
            x = test.index,
            y = test['close'],
            mode = 'lines',
            name = 'Ground Truth'
        )
        trace4 = go.Scatter(
            x = future_lstm.index,
            y = future_lstm['Predictions'],
            mode = 'lines',
            name = 'Future Price'
        )
        fig = go.Figure(data=[trace1, trace2, trace3, trace4])
        st.plotly_chart(fig)
    
        st.write("Find out more about the model's performance.")
        if st.button('Click me'):
            st.subheader("Evaluation Model")
            data_eval = pd.DataFrame({
                'Model': ['LSTM'],
                'Eval Loss Train-data': 0.0003,
                'Eval Loss Test-data': 0.0009,
                'MAE Train-data': 0.0099,
                'MAE Test-data': 0.0230,
                'R2-Score Train': 1.00,
                'R2-Score Test-data': 0.97
            })
            st.dataframe(data_eval)
            st.markdown('#')
            st.write("Conclusion from the evaluation results above!")
            st.write("The model we have created is slightly overfitting. But the model that we make is also able to make predictions well.")
    
    
    