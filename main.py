"""
Main Pipeline Script for Retail Market Basket Analysis
End-to-end execution of the market basket analysis project
"""

import os
import sys
from datetime import datetime

# Import all modules
from data_loader import load_data, load_sample_data
from preprocessing import preprocess_data, get_data_summary
from basket_transformation import prepare_basket_for_apriori
from apriori_analysis import run_apriori_pipeline
from visualization import generate_all_visualizations
from business_insights import generate_comprehensive_report, print_business_report, save_report_to_file


def main(use_sample_data=False, min_support=0.01, min_confidence=0.3, min_lift=1.0, max_items=100):
    """
    Main execution pipeline for Retail Market Basket Analysis
    
    Args:
        use_sample_data (bool): Whether to use sample data instead of Kaggle dataset
        min_support (float): Minimum support threshold for Apriori
        min_confidence (float): Minimum confidence threshold for rules
        min_lift (float): Minimum lift threshold for rules
        max_items (int): Maximum number of unique items to consider
    """
    print("=" * 70)
    print("RETAIL MARKET BASKET ANALYSIS - END-TO-END PIPELINE")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n[STEP 1/6] LOADING DATA")
    print("-" * 70)
    if use_sample_data:
        print("Using sample data for demonstration...")
        df = load_sample_data()
    else:
        print("Loading data from Kaggle...")
        try:
            df = load_data()
        except Exception as e:
            print(f"Error loading from Kaggle: {e}")
            print("Falling back to sample data...")
            df = load_sample_data()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Step 2: Preprocess Data
    print("\n[STEP 2/6] PREPROCESSING DATA")
    print("-" * 70)
    df_preprocessed = preprocess_data(df)
    
    # Get data summary
    summary = get_data_summary(df_preprocessed)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Step 3: Transform to Basket Format
    print("\n[STEP 3/6] TRANSFORMING TO BASKET FORMAT")
    print("-" * 70)
    basket_matrix = prepare_basket_for_apriori(
        df_preprocessed, 
        min_transaction_size=1, 
        max_items=max_items
    )
    
    # Step 4: Apply Apriori Algorithm
    print("\n[STEP 4/6] APPLYING APRIORI ALGORITHM")
    print("-" * 70)
    frequent_itemsets, rules, apriori_summary = run_apriori_pipeline(
        basket_matrix,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )
    
    if len(rules) == 0:
        print("\n⚠️  WARNING: No rules generated with current thresholds.")
        print("Try lowering the thresholds or using different parameters.")
        return
    
    # Step 5: Generate Visualizations
    print("\n[STEP 5/6] GENERATING VISUALIZATIONS")
    print("-" * 70)
    output_dir = './output'
    generate_all_visualizations(frequent_itemsets, rules, basket_matrix, output_dir)
    
    # Step 6: Generate Business Insights
    print("\n[STEP 6/6] GENERATING BUSINESS INSIGHTS")
    print("-" * 70)
    report = generate_comprehensive_report(rules, basket_matrix, frequent_itemsets)
    print_business_report(report)
    save_report_to_file(report, output_path=f'{output_dir}/business_insights_report.txt')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated Files:")
    print("  - top_rules_lift.png")
    print("  - top_rules_confidence.png")
    print("  - rules_heatmap.png")
    print("  - support_distribution.png")
    print("  - metrics_scatter.png")
    print("  - item_frequency.png")
    print("  - top_rules_summary.csv")
    print("  - business_insights_report.txt")
    print("=" * 70)


def run_with_custom_parameters():
    """
    Run the pipeline with custom parameters for different use cases
    """
    print("\nAvailable Analysis Profiles:")
    print("1. Conservative (High confidence, fewer rules)")
    print("2. Balanced (Moderate thresholds)")
    print("3. Exploratory (Low thresholds, more rules)")
    print("4. Custom (Enter your own parameters)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Conservative
        main(
            min_support=0.02,
            min_confidence=0.5,
            min_lift=1.5,
            max_items=50
        )
    elif choice == '2':
        # Balanced
        main(
            min_support=0.01,
            min_confidence=0.3,
            min_lift=1.0,
            max_items=100
        )
    elif choice == '3':
        # Exploratory
        main(
            min_support=0.005,
            min_confidence=0.2,
            min_lift=0.8,
            max_items=150
        )
    elif choice == '4':
        # Custom
        min_support = float(input("Enter min_support (0.001-0.1): "))
        min_confidence = float(input("Enter min_confidence (0.1-0.9): "))
        min_lift = float(input("Enter min_lift (0.5-5.0): "))
        max_items = int(input("Enter max_items (20-200): "))
        main(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_items=max_items
        )
    else:
        print("Invalid choice. Running with default parameters...")
        main()


if __name__ == "__main__":
    # Check if user wants to use sample data
    use_sample = True 
    
    # Check if user wants custom parameters
    custom = False 
    
    if custom:
        run_with_custom_parameters()
    else:
        # Run with default parameters
        main(use_sample_data=use_sample)
