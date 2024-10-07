from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Function to scale data
def scale_data(df, scaling_method):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif scaling_method == 'zscore':
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def clean_data(filepath, null_handling, scaling_method=None, custom_value=None):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    df = df.drop_duplicates()

    if null_handling == 'drop':
        df = df.dropna()
    elif null_handling == 'fill_mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif null_handling == 'fill_median':
        df = df.fillna(df.median(numeric_only=True))
    elif null_handling == 'fill_zero':
        df = df.fillna(0)
    elif null_handling == 'fill_custom' and custom_value is not None:
        try:
            custom_value = float(custom_value) if custom_value.isnumeric() else custom_value
        except ValueError:
            pass
        df = df.fillna(custom_value)
    elif null_handling == 'ffill':  # Forward fill
        df = df.ffill()
    elif null_handling == 'bfill':  # Backward fill
        df = df.bfill()

    df = remove_outliers(df)

    if scaling_method:
        df = scale_data(df, scaling_method)
    
    cleaned_filepath = filepath.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_filepath, index=False)

    return cleaned_filepath