"""
Apriori Algorithm Module for Retail Market Basket Analysis
Applies Apriori algorithm to discover frequent itemsets and association rules
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time


def apply_apriori(basket_matrix, min_support=0.01, use_colnames=True, max_len=None):
    """
    Apply Apriori algorithm to find frequent itemsets
    
    Args:
        basket_matrix (pd.DataFrame): Basket matrix (transactions x items)
        min_support (float): Minimum support threshold (0-1)
        use_colnames (bool): Use item names instead of column indices
        max_len (int): Maximum length of itemsets
        
    Returns:
        pd.DataFrame: Frequent itemsets with support values
    """
    print(f"Applying Apriori algorithm with min_support={min_support}...")
    start_time = time.time()
    
    # Apply Apriori
    frequent_itemsets = apriori(
        basket_matrix,
        min_support=min_support,
        use_colnames=use_colnames,
        max_len=max_len,
        low_memory=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Apriori completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    # Sort by support in descending order
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    # Add itemset length
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    return frequent_itemsets


def generate_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5):
    """
    Generate association rules from frequent itemsets
    
    Args:
        frequent_itemsets (pd.DataFrame): Frequent itemsets from Apriori
        metric (str): Metric to use for rule generation (confidence, lift, leverage, conviction)
        min_threshold (float): Minimum threshold for the metric
        
    Returns:
        pd.DataFrame: Association rules with various metrics
    """
    print(f"Generating association rules using {metric} >= {min_threshold}...")
    
    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )
    
    print(f"Generated {len(rules)} association rules")
    
    # Sort by lift in descending order (lift > 1 indicates positive association)
    rules = rules.sort_values('lift', ascending=False)
    
    return rules


def filter_rules_by_length(rules, min_antecedents=1, max_antecedents=3, min_consequents=1, max_consequents=2):
    """
    Filter rules by the length of antecedents and consequents
    
    Args:
        rules (pd.DataFrame): Association rules
        min_antecedents (int): Minimum number of items in antecedents
        max_antecedents (int): Maximum number of items in antecedents
        min_consequents (int): Minimum number of items in consequents
        max_consequents (int): Maximum number of items in consequents
        
    Returns:
        pd.DataFrame: Filtered rules
    """
    print(f"Filtering rules by length...")
    
    # Calculate lengths
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    # Filter
    filtered_rules = rules[
        (rules['antecedent_len'] >= min_antecedents) &
        (rules['antecedent_len'] <= max_antecedents) &
        (rules['consequent_len'] >= min_consequents) &
        (rules['consequent_len'] <= max_consequents)
    ]
    
    print(f"Rules after filtering: {len(filtered_rules)}")
    
    return filtered_rules


def filter_rules_by_metrics(rules, min_support=0.01, min_confidence=0.3, min_lift=1.0):
    """
    Filter rules by support, confidence, and lift thresholds
    
    Args:
        rules (pd.DataFrame): Association rules
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        
    Returns:
        pd.DataFrame: Filtered rules
    """
    print(f"Filtering rules by metrics...")
    
    filtered_rules = rules[
        (rules['support'] >= min_support) &
        (rules['confidence'] >= min_confidence) &
        (rules['lift'] >= min_lift)
    ]
    
    print(f"Rules after filtering: {len(filtered_rules)}")
    
    return filtered_rules


def get_top_rules(rules, n=20, sort_by='lift'):
    """
    Get top N rules sorted by a specified metric
    
    Args:
        rules (pd.DataFrame): Association rules
        n (int): Number of top rules to return
        sort_by (str): Metric to sort by (lift, confidence, support, leverage, conviction)
        
    Returns:
        pd.DataFrame: Top N rules
    """
    print(f"Getting top {n} rules sorted by {sort_by}...")
    
    top_rules = rules.sort_values(sort_by, ascending=False).head(n)
    
    return top_rules


def analyze_itemsets_by_length(frequent_itemsets):
    """
    Analyze frequent itemsets by their length
    
    Args:
        frequent_itemsets (pd.DataFrame): Frequent itemsets
        
    Returns:
        pd.DataFrame: Summary statistics by itemset length
    """
    print("Analyzing itemsets by length...")
    
    # Group by length
    length_stats = frequent_itemsets.groupby('length').agg({
        'support': ['count', 'mean', 'min', 'max'],
    }).round(4)
    
    length_stats.columns = ['count', 'mean_support', 'min_support', 'max_support']
    
    return length_stats


def get_itemset_summary(frequent_itemsets, rules):
    """
    Generate a comprehensive summary of itemsets and rules
    
    Args:
        frequent_itemsets (pd.DataFrame): Frequent itemsets
        rules (pd.DataFrame): Association rules
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'num_frequent_itemsets': len(frequent_itemsets),
        'num_association_rules': len(rules),
        'max_itemset_length': frequent_itemsets['length'].max(),
        'avg_itemset_length': frequent_itemsets['length'].mean(),
        'avg_support': frequent_itemsets['support'].mean(),
        'avg_confidence': rules['confidence'].mean() if len(rules) > 0 else 0,
        'avg_lift': rules['lift'].mean() if len(rules) > 0 else 0,
        'high_lift_rules': len(rules[rules['lift'] > 2]) if len(rules) > 0 else 0,
        'high_confidence_rules': len(rules[rules['confidence'] > 0.7]) if len(rules) > 0 else 0
    }
    
    return summary


def run_apriori_pipeline(basket_matrix, min_support=0.01, min_confidence=0.3, min_lift=1.0):
    """
    Complete pipeline for Apriori analysis
    
    Args:
        basket_matrix (pd.DataFrame): Basket matrix
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        
    Returns:
        tuple: (frequent_itemsets, rules, summary)
    """
    print("=" * 50)
    print("Running Apriori Analysis Pipeline")
    print("=" * 50)
    
    # Step 1: Apply Apriori
    frequent_itemsets = apply_apriori(basket_matrix, min_support=min_support)
    
    # Step 2: Generate association rules
    rules = generate_association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # Step 3: Filter rules by metrics
    rules = filter_rules_by_metrics(
        rules,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )
    
    # Step 4: Get summary
    summary = get_itemset_summary(frequent_itemsets, rules)
    
    print("=" * 50)
    print("Pipeline Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    return frequent_itemsets, rules, summary


if __name__ == "__main__":
    # Test Apriori analysis
    from data_loader import load_sample_data
    from preprocessing import preprocess_data
    from basket_transformation import prepare_basket_for_apriori
    
    df = load_sample_data()
    df_preprocessed = preprocess_data(df)
    basket_matrix = prepare_basket_for_apriori(df_preprocessed, max_items=20)
    
    # Run Apriori pipeline
    frequent_itemsets, rules, summary = run_apriori_pipeline(
        basket_matrix,
        min_support=0.1,
        min_confidence=0.3,
        min_lift=1.0
    )
    
    print("\nTop 5 Frequent Itemsets:")
    print(frequent_itemsets.head())
    
    if len(rules) > 0:
        print("\nTop 5 Rules:")
        print(rules.head())
    else:
        print("\nNo rules generated with current thresholds")
