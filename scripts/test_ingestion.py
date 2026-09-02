from src.data.ingestion import load_network_data
from src.data.validation import validate_network_data


FILE_PATH = "data/raw/network_metrics.csv"


def main():

    print("=" * 50)
    print("NETWORK DATA INGESTION TEST")
    print("=" * 50)

    print("\nLoading network data...")

    df = load_network_data(FILE_PATH)

    print("Network data loaded successfully!")

    print("\nValidating network data...")

    df = validate_network_data(df)

    print("Network data validated successfully!")

    print("\nDATA SUMMARY")
    print("-" * 50)

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")

    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()