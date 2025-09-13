import streamlit as st
import pandas as pd
import joblib


#load the trained pipeline 
#model = joblib.load("C:\Projects_ML\global-market-stock-prediction-and-dashboard\model\stacking_sr_model.pkl")

# --- Page Config ---
st.set_page_config(page_title="Global Stock Market Prediction", layout="wide")

# --- Background Gradient ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #00b09b, #006400);
    background-size: cover;
}

/* Change default text color */
body, p, div, h1, h2, h3, h4, h5, h6, label {
    color: black !important;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Page Title ---
st.title("🌍 Global Stock Market Prediction App")
st.write("Predict **daily stock index % change** using macroeconomic indicators.", )


# --- Sidebar for Input Features ---
st.sidebar.header("Input Features")

# Function to get user input
def user_input_features():
    # dropdowns
    country = st.sidebar.selectbox("Country", ["USA", "UK", "Germany", "China", "India"])
    stock_index = st.sidebar.text_input("Stock Index", "S&P 500")
    currency_code = st.sidebar.selectbox("Currency Code", ["USD", "EUR", "GBP", "CNY", "INR"])
    credit_rating = st.sidebar.selectbox("Credit Rating", ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"])
    
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
    banking_health = st.sidebar.slider("Banking Sector Health", 0, 100, 70)
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
# Example: Creating a new feature - Debt to Market Cap Ratio
input_df["Debt_to_Market_Cap"] = (input_df["Government_Debt_to_GDP_Percent"] * input_df["Market_Cap_Trillion_USD"]) / 100
input_df["Debt_to_Market_Cap"] = input_df["Debt_to_Market_Cap"].replace([float('inf'), -float('inf')], 0).fillna(0)

input_df["inflation_interest"] = input_df["Inflation_Rate_Percent"] * input_df["Interest_Rate_Percent"]
input_df["gdp_minus_unemp"] = input_df["GDP_Growth_Rate_Percent"] - input_df["Unemployment_Rate_Percent"]
input_df["oil_gold_ratio"] = input_df["Oil_Price_USD_Barrel"] / input_df["Gold_Price_USD_Ounce"]

# #scaling numerical features
# scaler = StandardScaler()
# num_features = ["Index_Value", "Market_Cap", "GDP_Growth", "Inflation", "Interest_Rate", "Unemployment", 
#                 "Exchange_Rate_to_USD", "Currency_Change_YTD", "Government_Debt_to_GDP", "Current_Account_Balance", 
#                 "FDI_Inflow", "Commodity_Index", "Oil_Price", "Gold_Price", "10Y_Bond_Yield", "Political_Risk_Score", 
#                 "Banking_Sector_Health", "Real_Estate_Index", "Export_Growth", "Import_Growth",
#                 "Debt_to_Market_Cap", "inflation_interest", "gdp_minus_unemp", "oil_gold_ratio"]
# input_df[num_features] = scaler.fit_transform(input_df[num_features])

# #encoding categorical features
# encoder = OneHotEncoder(drop = "first", sparse_output=False, handle_unknown='ignore')
# cat_features = ["Country", "Stock_Index", "Currency_Code", "Credit_Rating"]
# encoded_cat = pd.DataFrame(encoder.fit_transform(input_df[cat_features]), columns=encoder.get_feature_names_out(cat_features))
# input_data = pd.concat([input_df.drop(columns=cat_features), encoded_cat], axis=1)


# --- Prediction ---
# Uncomment the following lines when the model is available
# if st.button("Predict Stock Index Change"):
#     prediction = model.predict(input_data)[0]
#     st.subheader("📈 Prediction Result")
#     st.metric(label="Predicted Daily % Change", value=f"{prediction:.2f}%")
