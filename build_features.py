import pandas as pd
import numpy as np
import holidays
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time

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
    "Sydney, Australia": ( -33.87, 151.21),
    "Santo Domingo, República Dominicana": (18.48, -69.93),
    "San Salvador, El Salvador": (13.70, -89.20)
}

COUNTRY_CODE_MAP = {
    "Estados Unidos": "US",
    "Reino Unido": "GB",
    "Francia": "FR",
    "Alemania": "DE",
    "México": "MX",
    "Australia": "AU",
    "República Dominicana": "DO",
    "El Salvador": "SV"
}

def add_holiday_features(df):
    print("-> Adding holiday features...")
    country_holidays = {country: holidays.CountryHoliday(code) for country, code in COUNTRY_CODE_MAP.items()}
    def is_holiday(row):
        country = row["Order Country"]
        date = row["order_date_dt"]
        if country in country_holidays:
            return date in country_holidays[country]
        return False
    df['is_holiday'] = df.apply(is_holiday, axis=1).astype(int)
    return df

def add_weather_features(df):
    print("-> Adding weather features (this may take a few minutes)...")
    df['max_temp'] = np.nan
    df['precipitation'] = np.nan
    for location, group in df.groupby('order_location_key'):
        if location not in LOCATION_COORDINATE_CACHE:
            continue
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
            merged_group = pd.merge(group, df_weather, left_on='order_date_dt', right_on='time', how='left')
            df.loc[group.index, 'max_temp'] = merged_group['temperature_2m_max'].values
            df.loc[group.index, 'precipitation'] = merged_group['precipitation_sum'].values
        except Exception as e:
            print(f"  Could not fetch weather for {location}. Error: {e}")
        time.sleep(1)
    return df

def main():
    print("--- Hyperion Feature Engineering Pipeline Started ---")
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1', low_memory=False)

    print("-> Performing initial data cleaning...")
    
 
    df['order_date_dt'] = pd.to_datetime(df['order date (DateOrders)']).dt.normalize()
    
    df.sort_values('order_date_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['order_location_key'] = df['Order City'] + ", " + df['Order Country']

    df = add_holiday_features(df)
    print("-> SKIPPED: Economic feature step is suspended.")
    df = add_weather_features(df)

    print("-> Finalizing the dataset...")
    df.dropna(subset=['max_temp', 'precipitation'], inplace=True)
    
    final_columns = [
        'Type', 'Delivery Status', 'Category Name', 'Customer Country',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode',
        'Sales', 'Order Item Quantity',
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        'order_date_dt',
        'is_holiday',
        'max_temp',
        'precipitation'
    ]
    df_final = df[final_columns].copy()
    
    output_path = "hyperion_dataset.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Enriched dataset saved to: {output_path}")
    print(f"Total rows in final dataset: {len(df_final)}")
    print("Final columns:", df_final.columns.tolist())

if __name__ == "__main__":
    main()  