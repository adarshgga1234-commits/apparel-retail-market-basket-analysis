"""
Data Preprocessing Module for Retail Market Basket Analysis
Handles missing values, filters canceled transactions, and cleans data
"""

import pandas as pd
import numpy as np


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    print("Handling missing values...")
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    # Remove rows with missing CustomerID (critical for analysis)
    df_cleaned = df.dropna(subset=['CustomerID'])
    
    # Remove rows with missing Description
    df_cleaned = df_cleaned.dropna(subset=['Description'])
    
    # Remove rows with missing InvoiceNo
    df_cleaned = df_cleaned.dropna(subset=['InvoiceNo'])
    
    print(f"Missing values after cleaning:\n{df_cleaned.isnull().sum()}")
    print(f"Rows removed: {len(df) - len(df_cleaned)}")
    
    return df_cleaned


def filter_canceled_transactions(df):
    """
    Filter out canceled transactions (InvoiceNo starting with 'C')
    
    Args:
        df (pd.DataFrame): Dataset with missing values handled
        
    Returns:
        pd.DataFrame: Dataset with canceled transactions removed
    """
    print("Filtering canceled transactions...")
    
    # Convert InvoiceNo to string to check for 'C' prefix
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    
    # Count canceled transactions
    canceled_count = df[df['InvoiceNo'].str.startswith('C')].shape[0]
    print(f"Found {canceled_count} canceled transactions")
    
    # Filter out canceled transactions
    df_filtered = df[~df['InvoiceNo'].str.startswith('C')]
    
    print(f"Transactions after filtering: {len(df_filtered)}")
    
    return df_filtered


def filter_invalid_quantities(df):
    """
    Filter out transactions with invalid quantities (<= 0)
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with valid quantities only
    """
    print("Filtering invalid quantities...")
    
    invalid_count = df[df['Quantity'] <= 0].shape[0]
    print(f"Found {invalid_count} transactions with invalid quantities")
    
    df_filtered = df[df['Quantity'] > 0]
    
    print(f"Transactions after filtering: {len(df_filtered)}")
    
    return df_filtered


def filter_invalid_prices(df):
    """
    Filter out transactions with invalid unit prices (<= 0)
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with valid prices only
    """
    print("Filtering invalid prices...")
    
    invalid_count = df[df['UnitPrice'] <= 0].shape[0]
    print(f"Found {invalid_count} transactions with invalid prices")
    
    df_filtered = df[df['UnitPrice'] > 0]
    
    print(f"Transactions after filtering: {len(df_filtered)}")
    
    return df_filtered


def clean_descriptions(df):
    """
    Clean product descriptions (remove extra spaces, convert to uppercase)
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned descriptions
    """
    print("Cleaning product descriptions...")
    
    df['Description'] = df['Description'].str.strip().str.upper()
    
    return df


def add_derived_features(df):
    """
    Add derived features for analysis
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with derived features
    """
    print("Adding derived features...")
    
    # Calculate total amount per transaction
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Extract date and time from InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDay'] = df['InvoiceDate'].dt.day_name()
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    df['InvoiceMonth'] = df['InvoiceDate'].dt.month_name()
    
    return df


def preprocess_data(df):
    """
    Complete preprocessing pipeline
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Fully preprocessed dataset
    """
    print("=" * 50)
    print("Starting Data Preprocessing Pipeline")
    print("=" * 50)
    print(f"Initial dataset shape: {df.shape}")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    
    # Step 2: Filter canceled transactions
    df = filter_canceled_transactions(df)
    
    # Step 3: Filter invalid quantities
    df = filter_invalid_quantities(df)
    
    # Step 4: Filter invalid prices
    df = filter_invalid_prices(df)
    
    # Step 5: Clean descriptions
    df = clean_descriptions(df)
    
    # Step 6: Add derived features
    df = add_derived_features(df)
    
    print("=" * 50)
    print(f"Final dataset shape: {df.shape}")
    print(f"Data reduction: {((1 - len(df) / df.shape[0]) * 100):.2f}%")
    print("=" * 50)
    
    return df


def get_data_summary(df):
    """
    Generate a summary of the preprocessed data
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_transactions': df['InvoiceNo'].nunique(),
        'total_customers': df['CustomerID'].nunique(),
        'total_products': df['StockCode'].nunique(),
        'total_countries': df['Country'].nunique(),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'total_revenue': df['TotalAmount'].sum(),
        'avg_transaction_value': df.groupby('InvoiceNo')['TotalAmount'].sum().mean()
    }
    
    return summary


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_sample_data
    
    df = load_sample_data()
    df_preprocessed = preprocess_data(df)
    
    print("\nPreprocessed Data Sample:")
    print(df_preprocessed.head())
    
    summary = get_data_summary(df_preprocessed)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
