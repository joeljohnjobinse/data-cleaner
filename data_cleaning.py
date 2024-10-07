import pandas as pd
import sys
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def drop_duplicates(df):
    """
    Drops duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame.
    Compulsorily uses forward fill for missing values.
    """
    return df.fillna(method='ffill')

def remove_outliers(df, z_threshold=3):
    """
    Removes outliers from numerical columns in the DataFrame using the Z-score method.
    """
    numeric_cols = df.select_dtypes(include='number')
    z_scores = numeric_cols.apply(zscore).abs()
    df = df[(z_scores < z_threshold).all(axis=1)]
    return df

def scale_data(df, scaling_method='z_score'):
    """
    Scales numerical columns in the DataFrame using the specified method.
    :param scaling_method: Either 'z_score' or 'min_max'
    """
    numeric_cols = df.select_dtypes(include='number')

    if scaling_method == 'z_score':
        # Apply Z-score scaling
        df[numeric_cols.columns] = numeric_cols.apply(zscore)
    elif scaling_method == 'min_max':
        # Apply Min-Max scaling
        scaler = MinMaxScaler()
        df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)

    return df

def clean_data(input_path, output_path, scaling_method='z_score'):
    """
    Cleans the dataset by dropping duplicates, handling missing values,
    removing outliers, and scaling data.
    """
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop duplicates
    df = drop_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Remove outliers
    df = remove_outliers(df)

    # Scale data
    df = scale_data(df, scaling_method)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaning.py <input_path> <output_path> <scaling_method>")
        print("scaling_method: 'z_score' or 'min_max'")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scaling_method = sys.argv[3]

    # Clean the data
    clean_data(input_path, output_path, scaling_method)
