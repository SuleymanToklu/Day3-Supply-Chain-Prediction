# build_features.py

import pandas as pd
import numpy as np
import holidays
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time

# --- 1. LOCATION CACHE (The Smart Way to Handle Locations) ---
# In a real project, you would build this dynamically with a geocoding API like GeoPy,
# but to avoid API keys and slow processing, we'll use a pre-built cache for the most frequent locations.
LOCATION_COORDINATE_CACHE = {
    "Caguas, Puerto Rico": (18.23, -66.05),
    "Chicago, IL": (41.88, -87.63),
    "New York City, NY": (40.71, -74.01),
    "Los Angeles, CA": (34.05, -118.24),
    "Houston, TX": (29.76, -95.37),
    "London, UK": (51.51, -0.13),
    "Mexico City, Mexico": (19.43, -99.13),
    "Paris, France": (48.86, 2.35),
    "Berlin, Germany": (52.52, 13.41),
    "Sydney, Australia": ( -33.87, 151.21)
    # Add more key locations as needed
}

COUNTRY_CODE_MAP = {
    "Estados Unidos": "US",
    "Reino Unido": "GB",
    "Francia": "FR",
    "Alemania": "DE",
    "México": "MX",
    "Australia": "AU"
    # DataCo uses Spanish names. Map them to standard 2-letter codes.
}


# --- 2. FEATURE ENGINEERING FUNCTIONS ---

def add_holiday_features(df):
    """Adds a binary feature indicating if the order date was on a public holiday."""
    print("-> Adding holiday features...")
    
    # Create a dictionary of holiday objects for each country in our data
    country_holidays = {country: holidays.CountryHoliday(code) for country, code in COUNTRY_CODE_MAP.items()}
    
    def is_holiday(row):
        country = row["Order Country"]
        date = row["order_date_dt"]
        if country in country_holidays:
            return date in country_holidays[country]
        return False
        
    df['is_holiday'] = df.apply(is_holiday, axis=1).astype(int)
    return df

def add_economic_features(df):
    """Fetches and merges weekly average oil prices."""
    print("-> Adding economic features (Brent Oil Price)...")
    
    # Fetch data once
    brent_data = yf.download("BZ=F", start="2015-01-01", end="2018-02-01")
    brent_data.reset_index(inplace=True)
    brent_data = brent_data[['Date', 'Close']].rename(columns={'Close': 'oil_price_close'})
    
    # Use merge_asof to find the latest oil price for each order date
    df = pd.merge_asof(
        left=df.sort_values('order_date_dt'),
        right=brent_data.sort_values('Date'),
        left_on='order_date_dt',
        right_on='Date',
        direction='backward'
    )
    df.drop(columns=['Date'], inplace=True)
    return df

def add_weather_features(df):
    """Fetches and merges historical weather data for order locations."""
    print("-> Adding weather features (this may take a few minutes)...")
    df['max_temp'] = np.nan
    df['precipitation'] = np.nan

    # Group by location to make fewer, more efficient API calls
    for location, group in df.groupby('order_location_key'):
        if location not in LOCATION_COORDINATE_CACHE:
            continue # Skip locations not in our cache

        lat, lon = LOCATION_COORDINATE_CACHE[location]
        min_date = group['order_date_dt'].min().strftime('%Y-%m-%d')
        max_date = group['order_date_dt'].max().strftime('%Y-%m-%d')

        print(f"  Fetching weather for {location} from {min_date} to {max_date}...")

        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={min_date}&end_date={max_date}&daily=temperature_2m_max,precipitation_sum"
        
        try:
            response = requests.get(url)
            weather_data = response.json()['daily']
            df_weather = pd.DataFrame(weather_data)
            df_weather['time'] = pd.to_datetime(df_weather['time'])
            
            # Merge this weather data back into the main dataframe for this group
            merged_group = pd.merge(group, df_weather, left_on='order_date_dt', right_on='time', how='left')
            df.loc[group.index, 'max_temp'] = merged_group['temperature_2m_max'].values
            df.loc[group.index, 'precipitation'] = merged_group['precipitation_sum'].values

        except Exception as e:
            print(f"  Could not fetch weather for {location}. Error: {e}")
        
        time.sleep(1) # Be respectful to the API and avoid getting blocked

    return df

# --- 3. MAIN SCRIPT ---

def main():
    print("--- Hyperion Feature Engineering Pipeline Started ---")
    
    # Load base data
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1', low_memory=False)

    # Basic cleaning and preparation
    print("-> Performing initial data cleaning...")
    df['order_date_dt'] = pd.to_datetime(df['order date (DateOrders)'])
    df.sort_values('order_date_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Create a unique key for locations to use in our cache
    df['order_location_key'] = df['Order City'] + ", " + df['Order Country']

    # Apply feature engineering functions
    df = add_holiday_features(df)
    df = add_economic_features(df)
    df = add_weather_features(df)

    # Final cleanup
    print("-> Finalizing the dataset...")
    df.dropna(subset=['max_temp', 'precipitation', 'oil_price_close'], inplace=True)
    
    # Select final columns to save
    final_columns = [
        # Base features
        'Type', 'Delivery Status', 'Category Name', 'Customer Country',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode',
        'Sales', 'Order Item Quantity',
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        # New Hyperion features
        'order_date_dt',
        'is_holiday',
        'oil_price_close',
        'max_temp',
        'precipitation'
    ]
    df_final = df[final_columns].copy()
    
    # Save the enriched dataset
    output_path = "hyperion_dataset.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Enriched dataset saved to: {output_path}")
    print(f"Total rows in final dataset: {len(df_final)}")
    print("Final columns:", df_final.columns.tolist())

if __name__ == "__main__":
    main()