import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load sector data
auto_data = pd.read_csv('^CNXAUTO_historical_data.csv', parse_dates=True, index_col='Date')
bank_data = pd.read_csv('^NIFTYBANK_historical_data.csv', parse_dates=True, index_col='Date')
it_data = pd.read_csv('^CNXIT_historical_data.csv', parse_dates=True, index_col='Date')
pharma_data = pd.read_csv('^CNXPHARMA_historical_data.csv', parse_dates=True, index_col='Date')
energy_data = pd.read_csv('^CNXENERGY_historical_data.csv', parse_dates=True, index_col='Date')
finance_data = pd.read_csv('^CNXFINANCE_historical_data.csv', parse_dates=True, index_col='Date')

# Load macroeconomic data (example)
gdp_data = pd.read_csv('gdp_data.csv', parse_dates=True, index_col='DATE')
inflation_data = pd.read_csv('inflation_data.csv', parse_dates=True, index_col='DATE')
interest_rate_data = pd.read_csv('interest_rate_data.csv', parse_dates=True, index_col='DATE')

# Perform EDA
auto_data['Close'].plot(title='Nifty Auto Index')
plt.show()

gdp_data.plot(title='GDP Growth Rate')
plt.show()

# Combine data for correlation analysis
combined_data = auto_data[['Close']].join([gdp_data, inflation_data, interest_rate_data], how='inner')
print(combined_data.corr())

# Prepare data for modeling
combined_data['GDP'] = gdp_data.resample('M').mean()
combined_data['Inflation'] = inflation_data.resample('M').mean()
combined_data['InterestRate'] = interest_rate_data.resample('M').mean()
combined_data.dropna(inplace=True)

# Train-test split
train_data = combined_data.loc[:'2020']
test_data = combined_data.loc['2021':]

# ARIMA Model
try:
    model = ARIMA(train_data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    predictions_arima = model_fit.forecast(steps=len(test_data))
    
    # Evaluate ARIMA model
    mae_arima = mean_absolute_error(test_data['Close'], predictions_arima)
    mse_arima = mean_squared_error(test_data['Close'], predictions_arima)
    r2_arima = r2_score(test_data['Close'], predictions_arima)
    
    print(f'ARIMA - MAE: {mae_arima}, MSE: {mse_arima}, R2: {r2_arima}')
    
except Exception as e:
    print(f'Error in ARIMA model: {e}')

