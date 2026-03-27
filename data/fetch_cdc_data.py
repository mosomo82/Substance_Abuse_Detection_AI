import pandas as pd
import requests
from pathlib import Path

# The URL for the Provisional Overdose Death Counts
# We use a $limit=50000 to make sure we get more than the default 1000 rows
url = "https://data.cdc.gov/resource/xkb8-kh2a.json?$limit=50000"

print("Connecting to CDC database...")

try:
    # 1. Fetch the data
    response = requests.get(url)
    response.raise_for_status()  # Check if the download failed

    # 2. Convert to a Pandas DataFrame
    cdc_df = pd.DataFrame(response.json())

    # 3. Quick Check
    print(f"Success! Downloaded {len(cdc_df)} rows.")
    print("\nColumns available:")
    print(cdc_df.columns.tolist())

    # 4. Save it locally so you don't have to download it again
    # Resolve the output path relative to this script's location so the script
    # works regardless of which directory it is launched from.
    output_path = Path(__file__).parent / "raw" / "cdc_overdose_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # create data/raw/ if needed

    cdc_df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")

except Exception as e:
    print(f"Error fetching data: {e}")