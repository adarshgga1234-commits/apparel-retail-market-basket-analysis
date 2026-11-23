"""
Data Loader Module for Retail Market Basket Analysis
Handles downloading and loading the Online Retail dataset from Kaggle
"""

import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset_name='carrie1/online-retail', save_path='./data'):
    """
    Download the Online Retail dataset from Kaggle
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        save_path (str): Directory to save the dataset
        
    Returns:
        str: Path to the downloaded file
    """
    # Create data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle API authentication failed: {e}")
        print("Please ensure you have kaggle.json in ~/.kaggle/ directory")
        return None
    
    # Download dataset
    print(f"Downloading dataset: {dataset_name}")
    api.dataset_download_files(dataset_name, path=save_path, unzip=True)
    
    # Find the downloaded Excel file
    files = os.listdir(save_path)
    excel_files = [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')]
    
    if excel_files:
        return os.path.join(save_path, excel_files[0])
    else:
        print("No Excel file found after download")
        return None


def load_data(file_path=None):
    """
    Load the Online Retail dataset
    
    Args:
        file_path (str): Path to the dataset file. If None, will try to download from Kaggle
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path is None:
        file_path = download_kaggle_dataset()
        if file_path is None:
            raise FileNotFoundError("Could not download or find dataset")
    
    print(f"Loading data from: {file_path}")
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def load_sample_data():
    """
    Load a sample dataset for testing without Kaggle API
    Creates a sample retail dataset for demonstration
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    data = {
        'InvoiceNo': ['536365', '536365', '536365', '536366', '536366', '536367', '536367', '536368', '536368', '536369'],
        'StockCode': ['85123A', '71053', '84406B', '22423', '21755', '22197', '21212', '85099B', '22086', '22623'],
        'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS COAT HANGER', 'RECORD CAKE STAND', 'SET OF 3 CAKE TINS PANTRY DESIGN', 'SMALL POPCORN HOLDER', 'SET OF 3 CAKE TINS BUTTON DESIGN', 'JUMBO BAG RED RETROSPOT', 'CREAM CUPID HEARTS COAT HANGER', 'BOX OF 6 ASSORTED COLOUR TEASPOONS'],
        'Quantity': [6, 6, 8, 3, 6, 2, 3, 10, 6, 8],
        'InvoiceDate': pd.to_datetime(['2010-12-01 08:26:00', '2010-12-01 08:26:00', '2010-12-01 08:26:00', '2010-12-01 08:28:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00', '2010-12-01 08:35:00', '2010-12-01 08:35:00', '2010-12-01 08:36:00']),
        'UnitPrice': [2.55, 3.39, 2.75, 10.95, 4.95, 1.95, 4.95, 1.65, 2.75, 3.00],
        'CustomerID': [17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    }
    
    df = pd.DataFrame(data)
    print("Sample dataset created for testing")
    return df


if __name__ == "__main__":
    # Test loading
    try:
        df = load_data()
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Loading sample data instead...")
        df = load_sample_data()
        print(df.head())
