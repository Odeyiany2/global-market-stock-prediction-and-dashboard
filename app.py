import streamlit as st
import pandas as pd
import joblib
import os


#load the model and preprocessors
# import os
model = joblib.load(os.path.join("model and preprocessors", "stacking_sr_model.pkl"))
scaler = joblib.load(os.path.join("model and preprocessors", "scaler.pkl"))
encoder = joblib.load(os.path.join("model and preprocessors", "encoder.pkl"))

# --- Page Config ---
st.set_page_config(page_title="Global Stock Market Prediction", layout="wide")

# --- Background Gradient ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #00b09b, #006400);
    background-size: cover;
}

/* Change default text color for the main background only*/
[data-testid="stAppViewContainer"] .body,p,h1,h4, h2, h3, h5, h6 {
    color: black !important;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #00b09b, #006400);
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Page Title ---
st.title("🌍 Global Stock Market Prediction App 📈")
st.write("Predict **daily stock index % change** using macroeconomic indicators.", )


# --- Sidebar for Input Features ---
st.sidebar.header("🔍 Input Features")

# Function to get user input
def user_input_features():
    # dropdowns
    country = st.sidebar.selectbox("Country", [
        'United States', 'China', 'Japan', 'Germany', 'United Kingdom',
       'France', 'India', 'Canada', 'Brazil', 'Australia', 'South Korea',
       'Russia', 'Mexico', 'Italy', 'Spain', 'Netherlands', 'Switzerland',
       'Sweden', 'Norway', 'Denmark', 'Singapore', 'Hong Kong', 'Taiwan',
       'Indonesia', 'Thailand', 'Malaysia', 'Philippines', 'Vietnam',
       'Turkey', 'South Africa', 'Egypt', 'Nigeria', 'Chile', 'Argentina',
       'Colombia', 'Peru', 'UAE', 'Saudi Arabia', 'Israel'
    ])
    stock_index = st.sidebar.selectbox("Stock Index", [
        'S&P_500', 'Shanghai_Composite', 'Nikkei_225', 'DAX', 'FTSE_100',
       'CAC_40', 'Sensex', 'TSX', 'Bovespa', 'ASX_200', 'KOSPI', 'MOEX',
       'IPC', 'FTSE_MIB', 'IBEX_35', 'AEX', 'SMI', 'OMX_Stockholm', 'OSE',
       'OMXC_20', 'STI', 'Hang_Seng', 'TAIEX', 'JCI', 'SET', 'KLCI',
       'PSE', 'VN_Index', 'BIST_100', 'JSE', 'EGX_30', 'NSE', 'IPSA',
       'Merval', 'COLCAP', 'Lima_General', 'ADX', 'Tadawul', 'TA_125'
    ])
    currency_code = st.sidebar.selectbox("Currency Code", [
        'USD', 'CNY', 'JPY', 'EUR', 'GBP', 'INR', 'CAD', 'BRL', 'AUD',
       'KRW', 'RUB', 'MXN', 'CHF', 'SEK', 'NOK', 'DKK', 'SGD', 'HKD',
       'TWD', 'IDR', 'THB', 'MYR', 'PHP', 'VND', 'TRY', 'ZAR', 'EGP',
       'NGN', 'CLP', 'ARS', 'COP', 'PEN', 'AED', 'SAR', 'ILS'
    ])
    credit_rating = st.sidebar.selectbox("Credit Rating", [
        'AAA', 'A+', 'AA', 'BBB-', 'BB-', 'BB+', 'BBB', 'A', 'AA+', 'BBB+',
       'A-', 'B+', 'B-', 'CCC+'
    ])
    
    banking_health = st.sidebar.selectbox("Banking Sector Health", ['Strong', 'Moderate', 'Weak'])

    
    # Numeric inputs
    index_value = st.sidebar.number_input("Index Value", min_value=0.0, value=4000.0)
    market_cap = st.sidebar.number_input("Market Cap (Trillion USD)", min_value=0.0, value=25.0)
    gdp_growth = st.sidebar.number_input("GDP Growth Rate (%)", value=2.5)
    inflation = st.sidebar.number_input("Inflation Rate (%)", value=3.0)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", value=4.0)
    unemployment = st.sidebar.number_input("Unemployment Rate (%)", value=5.0)
    exchange_rate = st.sidebar.number_input("Exchange Rate to USD", value=1.0)
    currency_change = st.sidebar.number_input("Currency Change YTD (%)", value=0.5)
    govt_debt = st.sidebar.number_input("Government Debt (% of GDP)", value=60.0)
    current_account = st.sidebar.number_input("Current Account Balance (Billion USD)", value=100.0)
    fdi_inflow = st.sidebar.number_input("FDI Inflow (Billion USD)", value=50.0)
    commodity_index = st.sidebar.number_input("Commodity Index", value=120.0)
    oil_price = st.sidebar.number_input("Oil Price (USD/Barrel)", value=75.0)
    gold_price = st.sidebar.number_input("Gold Price (USD/Ounce)", value=1800.0)
    bond_yield = st.sidebar.number_input("10Y Bond Yield (%)", value=3.0)
    political_risk = st.sidebar.slider("Political Risk Score", 0, 100, 50)
    real_estate_index = st.sidebar.number_input("Real Estate Index", value=200.0)
    export_growth = st.sidebar.number_input("Export Growth (%)", value=2.0)
    import_growth = st.sidebar.number_input("Import Growth (%)", value=2.0)
    
    # --- Create Input DataFrame ---
    input_df = pd.DataFrame({
    'Country': [country],
    'Stock_Index': [stock_index],
    'Index_Value': [index_value],
    'Market_Cap_Trillion_USD': [market_cap],
    'GDP_Growth_Rate_Percent': [gdp_growth],
    'Inflation_Rate_Percent': [inflation],
    'Interest_Rate_Percent': [interest_rate],
    'Unemployment_Rate_Percent': [unemployment],
    'Currency_Code': [currency_code],
    'Exchange_Rate_USD': [exchange_rate],
    'Currency_Change_YTD_Percent': [currency_change],
    'Government_Debt_GDP_Percent': [govt_debt],
    'Current_Account_Balance_Billion_USD': [current_account],
    'FDI_Inflow_Billion_USD': [fdi_inflow],
    'Commodity_Index': [commodity_index],
    'Oil_Price_USD_Barrel': [oil_price],
    'Gold_Price_USD_Ounce': [gold_price],
    'Bond_Yield_10Y_Percent': [bond_yield],
    'Credit_Rating': [credit_rating],
    'Political_Risk_Score': [political_risk],
    'Banking_Sector_Health': [banking_health],
    'Real_Estate_Index': [real_estate_index],
    'Export_Growth_Percent': [export_growth],
    'Import_Growth_Percent': [import_growth]
})
    return input_df
input_df = user_input_features()

# --Feature Engineering---
input_df["inflation_interest"] = input_df["Inflation_Rate_Percent"] * input_df["Interest_Rate_Percent"]
input_df["gdp_minus_unemp"] = input_df["GDP_Growth_Rate_Percent"] - input_df["Unemployment_Rate_Percent"]
input_df["oil_gold_ratio"] = input_df["Oil_Price_USD_Barrel"] / input_df["Gold_Price_USD_Ounce"]

#scaling numerical features
num_features = ['Index_Value', 'Market_Cap_Trillion_USD', 'GDP_Growth_Rate_Percent', 'Inflation_Rate_Percent',
       'Interest_Rate_Percent', 'Unemployment_Rate_Percent', 'Exchange_Rate_USD',
       'Currency_Change_YTD_Percent', 'Government_Debt_GDP_Percent',
       'Current_Account_Balance_Billion_USD', 'FDI_Inflow_Billion_USD',
       'Commodity_Index', 'Oil_Price_USD_Barrel', 'Gold_Price_USD_Ounce',
       'Bond_Yield_10Y_Percent', 'Political_Risk_Score',
       'Real_Estate_Index', 'Export_Growth_Percent', 'Import_Growth_Percent',
       'inflation_interest', 'gdp_minus_unemp', 'oil_gold_ratio']
input_df[num_features] = scaler.transform(input_df[num_features])

#encoding categorical features
cat_features = ["Country", "Stock_Index", "Currency_Code", "Credit_Rating", "Banking_Sector_Health"]
encoded_cat = pd.DataFrame(
    encoder.transform(input_df[cat_features]),
    columns=encoder.get_feature_names_out(cat_features)
)
input_df = pd.concat([input_df.drop(columns=cat_features), encoded_cat], axis=1)


# --- Prediction ---
if st.sidebar.button("Predict Stock Index Change"):
    prediction = model.predict(input_df)[0]

    
    st.markdown("### 📈 Input Features used for Predictive Analysis")
    #show the input features 
    st.write(input_df)
    # Residual Error Band (approx using training RMSE if available)
    # assume ±0.5% as error margin -> just for demo use
    error_band = 0.5
    lower_bound = prediction - error_band
    upper_bound = prediction + error_band

    st.markdown("#### 📈 Prediction Result")
    st.markdown(f"##### The model predicts a daily stock index change of {prediction:.2f}%")
    st.write(f"Expected range: {lower_bound:.2f}% → {upper_bound:.2f}%")

    # # Display prediction interpretation
    # st.markdown(f"#### The model predicts a daily stock index change of **{prediction:.2f}%**.")
    #st.metric(label="**The Predicted Daily % Change is: **", value=f"{prediction:.2f}%")
    if prediction > 0:
        st.success("The stock index is predicted to rise 📈")
    elif prediction < 0:
        st.error("The stock index is predicted to fall 📉")
    else:
        st.info("The stock index is predicted to remain stable ⚖️")








# --- Footer ---st.markdown("---")
st.markdown("---")

st.markdown("Built by Miriam Itopa Odeyiany © 2025")
st.markdown("Find the project on [GitHub](https://github.com/Odeyiany2/global-market-stock-prediction-and-dashboard/tree/main)")