import pandas as pd

print("Analyzing the most frequent order locations...")

df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1', low_memory=False)

df['order_location_key'] = df['Order City'] + ", " + df['Order Country']

top_20_locations = df['order_location_key'].value_counts().head(20)

print("\n--- TOP 20 MOST FREQUENT ORDER LOCATIONS ---")
print(top_20_locations)
print("\nSuggestion: Pick a few of these, find their coordinates online, and add them to the cache.")