"""
Visualization Module for Retail Market Basket Analysis
Generates heatmaps, plots, and visual summaries of association rules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


# Set style
rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")


def plot_top_rules(rules, n=10, metric='lift', save_path=None):
    """
    Plot top N rules by a specified metric
    
    Args:
        rules (pd.DataFrame): Association rules
        n (int): Number of top rules to plot
        metric (str): Metric to sort by (lift, confidence, support)
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    print(f"Plotting top {n} rules by {metric}...")
    
    # Get top rules
    top_rules = rules.sort_values(metric, ascending=False).head(n).copy()
    
    # Convert frozensets to strings for display
    top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Create rule labels
    top_rules['rule'] = top_rules['antecedents_str'] + ' -> ' + top_rules['consequents_str']
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("viridis", n)
    bars = ax.barh(range(n), top_rules[metric].values, color=colors)
    
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_rules['rule'].values, fontsize=9)
    ax.set_xlabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n} Association Rules by {metric.capitalize()}', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_rules[metric].values)):
        ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_rules_heatmap(rules, n=20, save_path=None):
    """
    Create a heatmap showing relationships between items
    
    Args:
        rules (pd.DataFrame): Association rules
        n (int): Number of top rules to include
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    print(f"Creating heatmap for top {n} rules...")
    
    # Get top rules by lift
    top_rules = rules.sort_values('lift', ascending=False).head(n).copy()
    
    # Extract all unique items
    all_items = set()
    for idx, row in top_rules.iterrows():
        all_items.update(row['antecedents'])
        all_items.update(row['consequents'])
    
    all_items = list(all_items)
    
    # Create co-occurrence matrix
    co_occurrence = pd.DataFrame(0, index=all_items, columns=all_items)
    
    for idx, row in top_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # Add connections
        for ant in antecedents:
            for cons in consequents:
                co_occurrence.loc[ant, cons] += row['lift']
                co_occurrence.loc[cons, ant] += row['lift']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        co_occurrence,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Lift Value'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title('Item Association Heatmap (Top Rules by Lift)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Items', fontsize=12, fontweight='bold')
    ax.set_ylabel('Items', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    return fig


def plot_support_distribution(frequent_itemsets, save_path=None):
    """
    Plot distribution of support values for frequent itemsets
    
    Args:
        frequent_itemsets (pd.DataFrame): Frequent itemsets
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    print("Plotting support distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    axes[0].hist(frequent_itemsets['support'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Support Values', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by itemset length
    length_groups = frequent_itemsets.groupby('length')['support'].apply(list)
    axes[1].boxplot(length_groups.values, label=list(length_groups.index))
    axes[1].set_xlabel('Itemset Length', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Support', fontsize=12, fontweight='bold')
    axes[1].set_title('Support by Itemset Length', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Support distribution plot saved to {save_path}")
    
    return fig


def plot_metrics_scatter(rules, save_path=None):
    """
    Create scatter plot showing relationship between different metrics
    
    Args:
        rules (pd.DataFrame): Association rules
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    print("Creating metrics scatter plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Support vs Confidence
    scatter1 = axes[0].scatter(rules['support'], rules['confidence'], 
                               c=rules['lift'], cmap='viridis', alpha=0.6, s=50)
    axes[0].set_xlabel('Support', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Confidence', fontsize=11, fontweight='bold')
    axes[0].set_title('Support vs Confidence', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Lift')
    
    # Support vs Lift
    scatter2 = axes[1].scatter(rules['support'], rules['lift'], 
                               c=rules['confidence'], cmap='plasma', alpha=0.6, s=50)
    axes[1].set_xlabel('Support', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Lift', fontsize=11, fontweight='bold')
    axes[1].set_title('Support vs Lift', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Confidence')
    
    # Confidence vs Lift
    scatter3 = axes[2].scatter(rules['confidence'], rules['lift'], 
                               c=rules['support'], cmap='coolwarm', alpha=0.6, s=50)
    axes[2].set_xlabel('Confidence', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Lift', fontsize=11, fontweight='bold')
    axes[2].set_title('Confidence vs Lift', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[2], label='Support')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics scatter plot saved to {save_path}")
    
    return fig


def plot_item_frequency(basket_matrix, top_n=20, save_path=None):
    """
    Plot frequency of top N items
    
    Args:
        basket_matrix (pd.DataFrame): Basket matrix
        top_n (int): Number of top items to display
        save_path (str): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    print(f"Plotting top {top_n} item frequencies...")
    
    # Calculate item frequencies
    item_freq = basket_matrix.sum(axis=0).sort_values(ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("rocket", top_n)
    bars = ax.barh(range(len(item_freq)), item_freq.values, color=colors)
    
    ax.set_yticks(range(len(item_freq)))
    ax.set_yticklabels(item_freq.index, fontsize=10)
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Frequent Items', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, item_freq.values)):
        ax.text(value + max(item_freq.values) * 0.01, i, f'{int(value)}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Item frequency plot saved to {save_path}")
    
    return fig


def create_summary_table(rules, n=20, save_path=None):
    """
    Create a formatted summary table of top rules
    
    Args:
        rules (pd.DataFrame): Association rules
        n (int): Number of top rules to include
        save_path (str): Path to save the table (CSV)
        
    Returns:
        pd.DataFrame: Summary table
    """
    print(f"Creating summary table for top {n} rules...")
    
    # Get top rules by lift
    top_rules = rules.sort_values('lift', ascending=False).head(n).copy()
    
    # Convert frozensets to strings
    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Select and rename columns
    summary_table = top_rules[[
        'antecedents', 'consequents', 'support', 'confidence', 'lift'
    ]].copy()
    
    # Round numeric columns
    summary_table['support'] = summary_table['support'].round(4)
    summary_table['confidence'] = summary_table['confidence'].round(4)
    summary_table['lift'] = summary_table['lift'].round(4)
    
    # Add rule number
    summary_table.insert(0, 'Rule_ID', range(1, len(summary_table) + 1))
    
    # Rename columns for better display
    summary_table.columns = [
        'Rule ID', 'Antecedents (If)', 'Consequents (Then)', 
        'Support', 'Confidence', 'Lift'
    ]
    
    if save_path:
        summary_table.to_csv(save_path, index=False)
        print(f"Summary table saved to {save_path}")
    
    return summary_table


def generate_all_visualizations(frequent_itemsets, rules, basket_matrix, output_dir='./output'):
    """
    Generate all visualizations and save them
    
    Args:
        frequent_itemsets (pd.DataFrame): Frequent itemsets
        rules (pd.DataFrame): Association rules
        basket_matrix (pd.DataFrame): Basket matrix
        output_dir (str): Directory to save visualizations
    """
    import os
    
    print("=" * 50)
    print("Generating All Visualizations")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    plot_top_rules(rules, n=15, metric='lift', 
                   save_path=f'{output_dir}/top_rules_lift.png')
    
    plot_top_rules(rules, n=15, metric='confidence', 
                   save_path=f'{output_dir}/top_rules_confidence.png')
    
    plot_rules_heatmap(rules, n=15, 
                      save_path=f'{output_dir}/rules_heatmap.png')
    
    plot_support_distribution(frequent_itemsets, 
                              save_path=f'{output_dir}/support_distribution.png')
    
    plot_metrics_scatter(rules, 
                        save_path=f'{output_dir}/metrics_scatter.png')
    
    plot_item_frequency(basket_matrix, top_n=20, 
                       save_path=f'{output_dir}/item_frequency.png')
    
    # Create summary table
    summary_table = create_summary_table(rules, n=20, 
                                         save_path=f'{output_dir}/top_rules_summary.csv')
    
    print("\nSummary Table:")
    print(summary_table.to_string(index=False))
    
    print("=" * 50)
    print(f"All visualizations saved to {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    # Test visualization
    from data_loader import load_sample_data
    from preprocessing import preprocess_data
    from basket_transformation import prepare_basket_for_apriori
    from apriori_analysis import run_apriori_pipeline
    
    df = load_sample_data()
    df_preprocessed = preprocess_data(df)
    basket_matrix = prepare_basket_for_apriori(df_preprocessed, max_items=20)
    
    frequent_itemsets, rules, summary = run_apriori_pipeline(
        basket_matrix,
        min_support=0.1,
        min_confidence=0.3,
        min_lift=1.0
    )
    
    if len(rules) > 0:
        generate_all_visualizations(frequent_itemsets, rules, basket_matrix)
    else:
        print("No rules to visualize")
