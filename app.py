import streamlit as st
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from babel.numbers import format_currency

# --- Sidebar Configuration ---
st.sidebar.header("Gold Price Predictor Settings")
usd_to_inr = st.sidebar.number_input("USD to INR Conversion Rate", min_value=1.0, value=80.0, step=0.5)
local_premium = st.sidebar.number_input("Local Premium Factor", min_value=1.0, value=1.57, step=0.1)

# --- Main App Title ---
st.title("Prasun's Gold Price Prediction (India)")

# --- Date Input Widget ---
selected_date = st.date_input("Select Date", value=pd.Timestamp.today())

# --- Data Download Function ---
@st.cache_data
def get_data(end_date):
    """Fetch gold price data up to the specified end_date."""
    data = yf.download('GC=F', start='2010-01-01', end=end_date)
    if 'Close' not in data.columns:
        return pd.DataFrame()
    return data[['Close']].reset_index()

# --- Data Preprocessing Function ---
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    for i in range(1, 31):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# --- Model Training Function ---
@st.cache_resource
def train_model(df):
    X = df.drop('Close', axis=1)
    y = df['Close']
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2

# --- Main Application ---
if st.button("Predict"):
    with st.spinner("Fetching data and training model..."):
        end_date = pd.to_datetime(selected_date).strftime('%Y-%m-%d')
        raw_data = get_data(end_date)
        if raw_data.empty:
            st.error(f"No data available up to {end_date}.")
        else:
            processed_data = preprocess_data(raw_data)
            if processed_data.empty:
                st.error("Not enough data to generate features for prediction.")
            else:
                latest_features = processed_data.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
                model, mae, r2 = train_model(processed_data)
                prediction_usd_per_ounce = model.predict(latest_features)[0]
                prediction_inr_per_ounce = prediction_usd_per_ounce * usd_to_inr * local_premium
                formatted_price_per_ounce = format_currency(round(prediction_inr_per_ounce, 2), 'INR', locale='en_IN')
    
    # --- Display Results ---
    st.success(f"**Predicted Gold Price on {end_date}:**")
    st.write(f"**Price per ounce:** {formatted_price_per_ounce}")
    st.markdown("---")
    st.info(f"**Model Accuracy Metrics:** MAE = {mae:.2f} USD, R² = {r2:.2f}")
