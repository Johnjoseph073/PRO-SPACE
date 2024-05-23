import yfinance as yf
import pandas as pd

# Define the list of sector indices
sectors = {
    '^CNXAUTO': 'Nifty Auto',
    '^NIFTYBANK': 'Nifty Bank',
    '^CNXIT': 'Nifty IT',
    '^CNXPHARMA': 'Nifty Pharma',
    '^CNXENERGY': 'Nifty Energy',
    '^CNXFINANCE': 'Nifty Finance'
}

# Download the last 10 years of data for each sector
sector_data = {}
for sector, name in sectors.items():
    try:
        data = yf.download(sector, start='2013-01-01', end='2023-01-01')
        if data.empty:
            raise ValueError(f"No data found for {sector}")
        sector_data[sector] = data
        data.to_csv(f'{sector}_historical_data.csv')
        print(f"Successfully downloaded data for {name}")
    except Exception as e:
        print(f"Failed to download data for {name}: {e}")
