"""
Basket Transformation Module for Retail Market Basket Analysis
Transforms transactional data into basket-format dataframe for association rule mining
"""

import pandas as pd
import numpy as np
from collections import Counter


def create_basket_matrix(df, group_by='InvoiceNo', item_col='Description'):
    """
    Transform transactional data into basket-format matrix
    
    Args:
        df (pd.DataFrame): Preprocessed transactional data
        group_by (str): Column to group by (default: InvoiceNo)
        item_col (str): Column containing item names (default: Description)
        
    Returns:
        pd.DataFrame: Basket matrix where rows are transactions and columns are items
    """
    print("Creating basket matrix...")
    
    # Group by invoice and collect items
    basket = df.groupby([group_by])[item_col].apply(lambda x: ','.join(x)).reset_index()
    
    # Create one-hot encoded matrix
    basket_items = basket[item_col].str.get_dummies(sep=',')
    
    # Add the group_by column back
    basket_matrix = pd.concat([basket[group_by], basket_items], axis=1)
    
    print(f"Basket matrix shape: {basket_matrix.shape}")
    print(f"Number of transactions: {basket_matrix.shape[0]}")
    print(f"Number of unique items: {basket_matrix.shape[1] - 1}")
    
    return basket_matrix


def create_basket_matrix_by_customer(df, customer_col='CustomerID', item_col='Description'):
    """
    Create basket matrix grouped by customer instead of invoice
    
    Args:
        df (pd.DataFrame): Preprocessed transactional data
        customer_col (str): Column containing customer IDs
        item_col (str): Column containing item names
        
    Returns:
        pd.DataFrame: Basket matrix where rows are customers and columns are items
    """
    print("Creating customer basket matrix...")
    
    # Group by customer and collect items
    basket = df.groupby([customer_col])[item_col].apply(lambda x: ','.join(x)).reset_index()
    
    # Create one-hot encoded matrix
    basket_items = basket[item_col].str.get_dummies(sep=',')
    
    # Add the customer column back
    basket_matrix = pd.concat([basket[customer_col], basket_items], axis=1)
    
    print(f"Customer basket matrix shape: {basket_matrix.shape}")
    print(f"Number of customers: {basket_matrix.shape[0]}")
    print(f"Number of unique items: {basket_matrix.shape[1] - 1}")
    
    return basket_matrix


def filter_by_country(df, country='United Kingdom'):
    """
    Filter data by specific country
    
    Args:
        df (pd.DataFrame): Preprocessed data
        country (str): Country name to filter by
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    print(f"Filtering data for country: {country}")
    
    df_filtered = df[df['Country'] == country].copy()
    
    print(f"Transactions after filtering: {len(df_filtered)}")
    
    return df_filtered


def filter_top_items(df, item_col='Description', top_n=100):
    """
    Filter to keep only top N most frequent items
    
    Args:
        df (pd.DataFrame): Preprocessed data
        item_col (str): Column containing item names
        top_n (int): Number of top items to keep
        
    Returns:
        pd.DataFrame: Filtered dataset with top items only
    """
    print(f"Filtering to top {top_n} items...")
    
    # Get item frequencies
    item_counts = df[item_col].value_counts()
    
    # Get top N items
    top_items = item_counts.head(top_n).index.tolist()
    
    # Filter dataset
    df_filtered = df[df[item_col].isin(top_items)].copy()
    
    print(f"Transactions after filtering: {len(df_filtered)}")
    print(f"Items retained: {len(top_items)}")
    
    return df_filtered


def create_basket_with_quantity(df, group_by='InvoiceNo', item_col='Description', quantity_col='Quantity'):
    """
    Create basket matrix with quantities instead of binary values
    
    Args:
        df (pd.DataFrame): Preprocessed transactional data
        group_by (str): Column to group by
        item_col (str): Column containing item names
        quantity_col (str): Column containing quantities
        
    Returns:
        pd.DataFrame: Basket matrix with quantities
    """
    print("Creating basket matrix with quantities...")
    
    # Pivot the data
    basket = df.pivot_table(
        index=group_by,
        columns=item_col,
        values=quantity_col,
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    print(f"Basket matrix shape: {basket.shape}")
    
    return basket


def get_basket_statistics(basket_matrix):
    """
    Get statistics about the basket matrix
    
    Args:
        basket_matrix (pd.DataFrame): Basket matrix
        
    Returns:
        dict: Statistics
    """
    # Remove the group_by column for calculations
    item_columns = basket_matrix.columns[1:] if basket_matrix.columns[0] in ['InvoiceNo', 'CustomerID'] else basket_matrix.columns
    
    stats = {
        'num_transactions': basket_matrix.shape[0],
        'num_items': len(item_columns),
        'avg_items_per_transaction': basket_matrix[item_columns].sum(axis=1).mean(),
        'max_items_in_transaction': basket_matrix[item_columns].sum(axis=1).max(),
        'min_items_in_transaction': basket_matrix[item_columns].sum(axis=1).min(),
        'most_frequent_items': basket_matrix[item_columns].sum().sort_values(ascending=False).head(10).to_dict()
    }
    
    return stats


def prepare_basket_for_apriori(df, min_transaction_size=1, max_items=100):
    """
    Complete pipeline to prepare basket data for Apriori algorithm
    
    Args:
        df (pd.DataFrame): Preprocessed transactional data
        min_transaction_size (int): Minimum number of items per transaction
        max_items (int): Maximum number of unique items to consider
        
    Returns:
        pd.DataFrame: Prepared basket matrix
    """
    print("=" * 50)
    print("Preparing Basket Data for Apriori Algorithm")
    print("=" * 50)
    
    # Filter by country (UK has most transactions)
    df = filter_by_country(df, country='United Kingdom')
    
    # Filter to top items for performance
    df = filter_top_items(df, top_n=max_items)
    
    # Create basket matrix
    basket_matrix = create_basket_matrix(df)
    
    # Remove the InvoiceNo column for Apriori
    basket_for_apriori = basket_matrix.iloc[:, 1:]
    
    # Filter transactions with minimum size
    transaction_sizes = basket_for_apriori.sum(axis=1)
    valid_transactions = transaction_sizes >= min_transaction_size
    
    basket_final = basket_for_apriori[valid_transactions]
    
    print(f"Final basket shape for Apriori: {basket_final.shape}")
    print(f"Transactions with >= {min_transaction_size} items: {basket_final.shape[0]}")
    print("=" * 50)
    
    return basket_final


if __name__ == "__main__":
    # Test basket transformation
    from data_loader import load_sample_data
    from preprocessing import preprocess_data
    
    df = load_sample_data()
    df_preprocessed = preprocess_data(df)
    
    # Create basket matrix
    basket_matrix = create_basket_matrix(df_preprocessed)
    print("\nBasket Matrix Sample:")
    print(basket_matrix.head())
    
    # Get statistics
    stats = get_basket_statistics(basket_matrix)
    print("\nBasket Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
