import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

path = '/Users/rohanjairam/projects/financial-fraud-anomaly-detector/data/creditcard.csv'

df = pd.read_csv(path)

def load_and_clean_data(df):
    # Handle missing values (if any)
    df = df.dropna()
    # Scale Amount and Time if present
    scaler = StandardScaler()
    for col in ['Amount', 'Time']:
        if col in df.columns:
            df[f'scaled_{col.lower()}'] = scaler.fit_transform(df[[col]])
            df = df.drop(columns=[col])
    return df

def add_engineered_features(df):
    # For creditcard.csv, apply directly (no 'account' column)
    if 'scaled_time' in df.columns:
        df['time_diff'] = df['scaled_time'].diff().fillna(0)
    if 'scaled_amount' in df.columns:
        df['rolling_mean_amount'] = df['scaled_amount'].rolling(window=5).mean().fillna(0)
    return df

def add_basic_features(df):
    # Example: Amount * V1 interaction
    if 'scaled_amount' in df.columns and 'V1' in df.columns:
        df['v_amount_interaction'] = df['scaled_amount'] * df['V1']
    return df

def splitdata(df):
    x = df.drop(columns=['Class'])
    y = df['Class']
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

df = load_and_clean_data(df)
df = add_engineered_features(df)
df = add_basic_features(df)
df.to_csv('/Users/rohanjairam/projects/financial-fraud-anomaly-detector/data/updated_creditcard.csv', index=False)