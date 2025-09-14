# **📈 Predicting Global Stock Market Performance Using Macroeconomic Indicators**

## 📌 **Project Overview**

This project analyzes global financial and macroeconomic data across 39 countries to understand the relationships between stock indices, economic indicators, and risk factors.

Using Machine Learning (Stacking Regressor), the project predicts daily stock index percentage changes based on features such as GDP growth, inflation, interest rates, unemployment, commodity prices, and political risk scores.

It also includes:

* An interactive Streamlit app where users can input macroeconomic indicators and get stock market predictions.

* A Power BI dashboard for financial storytelling, allowing dynamic exploration of global markets, risks, and stability.

## 🔎 **Objectives**

 * Analyze how macroeconomic and risk indicators influence stock indices.

* Build machine learning models to predict stock market performance.

* Create a dashboard to visualize insights for investors and policymakers.

## **🛠️ Tech Stack**

* Languages: Python, SQL

* Libraries: pandas, scikit-learn, LightGBM, joblib, matplotlib, seaborn

* Deployment: Streamlit

* Visualization: Power BI

* Version Control: Git/GitHub

## **📊 Key Features**

* Data cleaning, feature engineering, and model training for predictive analytics

* Stock market prediction using an ensemble Stacking Regressor (achieved R² ≈ 0.8)

* Exploratory Data Analysis with insights into GDP, inflation, and trade dynamics

* Streamlit App: Interactive prediction tool with user input & visualization

* Power BI Dashboard:

   * Market Cap rankings (Top 10 countries)

   * GDP vs Inflation scatter plots

   * Political map

   * Stock Index by Index Value

## **📈 Results**

* Achieved R² score of 0.8 on test data using a Stacking Regressor.

* Built a dual-deployment solution (Streamlit + Power BI) for both prediction and storytelling.

* Insights: Political risk, interest rates, and GDP growth showed strong correlations with stock market volatility.

## **🚀 How to Run Locally**
* Clone the repo
  ```
  git clone https://github.com/your-username/global-market-stock-prediction.git
  cd global-market-stock-prediction
  ```
* Install Dependencies
  ```
  pip install -r requirements.txt
  ```
* Run the Streamlit App
  ```
  streamlit run app.py
  ```

## ** 🚀 Deployments**
* PowerBI
  
* Streamlit App

  
## **🙌 Acknowledgements**

* Dataset sourced from Kaggle: [Global Finance Data 2024](https://www.kaggle.com/datasets/imaadmahmood/global-finance-and-economic-indicators-dataset-2024)

* Inspired by the intersection of Data Science, Finance, and Risk Management.
