# Retail Market Basket Analysis

An end-to-end data engineering project that applies the Apriori algorithm to discover frequent itemsets and association rules in retail transaction data. The project includes data preprocessing, basket transformation, association rule mining, visualization, and actionable business insights for store layout optimization.

## 📊 Project Overview

This project analyzes the Online Retail dataset from Kaggle to:
- Load and preprocess transactional data
- Handle missing values and filter canceled transactions
- Transform data into basket format for analysis
- Apply the Apriori algorithm to discover frequent itemsets
- Generate association rules with support, confidence, and lift metrics
- Visualize relationships using heatmaps and plots
- Formulate actionable business insights for store layout optimization

## 🏗️ Project Structure

```
data ware/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data_loader.py                     # Data loading module
├── preprocessing.py                   # Data preprocessing module
├── basket_transformation.py          # Basket transformation module
├── apriori_analysis.py                # Apriori algorithm module
├── visualization.py                   # Visualization module
├── business_insights.py               # Business insights module
├── main.py                            # Main pipeline script
└── output/                            # Generated outputs (created after running)
    ├── top_rules_lift.png
    ├── top_rules_confidence.png
    ├── rules_heatmap.png
    ├── support_distribution.png
    ├── metrics_scatter.png
    ├── item_frequency.png
    ├── top_rules_summary.csv
    └── business_insights_report.txt
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Kaggle API credentials (for downloading the dataset)

### Setup

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API credentials:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Create a new API token
   - Place `kaggle.json` in `~/.kaggle/` directory (or set `KAGGLE_CONFIG_DIR` environment variable)

## 📖 Usage

### Quick Start

Run the main pipeline with default parameters:

```bash
python main.py
```

You will be prompted to:
1. Choose whether to use sample data (for testing) or download from Kaggle
2. Choose analysis profile (Conservative, Balanced, Exploratory, or Custom)

### Running with Sample Data

For quick testing without Kaggle API:

```python
from main import main

main(use_sample_data=True)
```

### Custom Parameters

Run with custom thresholds:

```python
from main import main

main(
    use_sample_data=False,
    min_support=0.01,      # Minimum support threshold (0-1)
    min_confidence=0.3,    # Minimum confidence threshold (0-1)
    min_lift=1.0,          # Minimum lift threshold
    max_items=100          # Maximum number of unique items to consider
)
```

### Individual Module Usage

Each module can be used independently:

```python
# Load data
from data_loader import load_data
df = load_data()

# Preprocess data
from preprocessing import preprocess_data
df_clean = preprocess_data(df)

# Transform to basket format
from basket_transformation import prepare_basket_for_apriori
basket = prepare_basket_for_apriori(df_clean, max_items=100)

# Apply Apriori
from apriori_analysis import run_apriori_pipeline
frequent_itemsets, rules, summary = run_apriori_pipeline(
    basket,
    min_support=0.01,
    min_confidence=0.3,
    min_lift=1.0
)

# Generate visualizations
from visualization import generate_all_visualizations
generate_all_visualizations(frequent_itemsets, rules, basket)

# Generate business insights
from business_insights import generate_comprehensive_report, print_business_report
report = generate_comprehensive_report(rules, basket, frequent_itemsets)
print_business_report(report)
```

## 📊 Module Descriptions

### data_loader.py
Handles downloading and loading the Online Retail dataset from Kaggle. Includes a sample data generator for testing without API access.

**Key Functions:**
- `download_kaggle_dataset()` - Downloads dataset from Kaggle
- `load_data()` - Loads the dataset from file
- `load_sample_data()` - Creates sample data for testing

### preprocessing.py
Performs data cleaning and preprocessing operations.

**Key Functions:**
- `handle_missing_values()` - Removes rows with missing critical fields
- `filter_canceled_transactions()` - Removes transactions with InvoiceNo starting with 'C'
- `filter_invalid_quantities()` - Removes transactions with quantity <= 0
- `filter_invalid_prices()` - Removes transactions with unit price <= 0
- `preprocess_data()` - Complete preprocessing pipeline

### basket_transformation.py
Transforms transactional data into basket format for association rule mining.

**Key Functions:**
- `create_basket_matrix()` - Creates one-hot encoded basket matrix
- `filter_by_country()` - Filters data by country
- `filter_top_items()` - Keeps only top N most frequent items
- `prepare_basket_for_apriori()` - Complete basket preparation pipeline

### apriori_analysis.py
Applies the Apriori algorithm to discover frequent itemsets and association rules.

**Key Functions:**
- `apply_apriori()` - Runs Apriori algorithm
- `generate_association_rules()` - Generates association rules
- `filter_rules_by_metrics()` - Filters rules by support, confidence, lift
- `run_apriori_pipeline()` - Complete Apriori analysis pipeline

### visualization.py
Generates various visualizations for the analysis results.

**Key Functions:**
- `plot_top_rules()` - Bar chart of top rules by metric
- `plot_rules_heatmap()` - Heatmap showing item associations
- `plot_support_distribution()` - Distribution of support values
- `plot_metrics_scatter()` - Scatter plots of rule metrics
- `plot_item_frequency()` - Most frequent items
- `create_summary_table()` - CSV summary of top rules

### business_insights.py
Generates actionable business insights from the analysis results.

**Key Functions:**
- `analyze_cross_selling_opportunities()` - Identifies cross-selling opportunities
- `generate_store_layout_recommendations()` - Store layout suggestions
- `generate_product_bundling_suggestions()` - Product bundling ideas
- `generate_promotional_strategies()` - Promotional strategy recommendations
- `generate_comprehensive_report()` - Complete business insights report

## 📈 Analysis Profiles

### Conservative (High Confidence)
- `min_support: 0.02`
- `min_confidence: 0.5`
- `min_lift: 1.5`
- `max_items: 50`
- **Use case:** When you need highly reliable rules with strong associations

### Balanced (Default)
- `min_support: 0.01`
- `min_confidence: 0.3`
- `min_lift: 1.0`
- `max_items: 100`
- **Use case:** Good balance between rule quantity and quality

### Exploratory (Low Thresholds)
- `min_support: 0.005`
- `min_confidence: 0.2`
- `min_lift: 0.8`
- `max_items: 150`
- **Use case:** Discovering all possible associations, including weaker ones

## 📊 Metrics Explained

### Support
The proportion of transactions that contain the itemset.
- Range: 0 to 1
- Higher support = more frequent itemset

### Confidence
The conditional probability that a transaction containing the antecedent also contains the consequent.
- Range: 0 to 1
- Higher confidence = stronger rule

### Lift
The ratio of the observed support to the expected support if the items were independent.
- Lift > 1: Positive association (items appear together more often than expected)
- Lift = 1: No association (items are independent)
- Lift < 1: Negative association (items appear together less often than expected)

## 🎯 Business Insights Generated

The project generates actionable insights including:

1. **Cross-Selling Opportunities** - Items frequently bought together
2. **Store Layout Recommendations** - Product placement suggestions
3. **Product Bundling Suggestions** - Bundle deals for promotions
4. **Category Affinity Analysis** - Relationships between product categories
5. **Promotional Strategies** - Marketing and promotion recommendations

## 📝 Output Files

After running the pipeline, the following files are generated in the `output/` directory:

- `top_rules_lift.png` - Bar chart of top rules by lift
- `top_rules_confidence.png` - Bar chart of top rules by confidence
- `rules_heatmap.png` - Heatmap showing item associations
- `support_distribution.png` - Distribution of support values
- `metrics_scatter.png` - Scatter plots of rule metrics
- `item_frequency.png` - Most frequent items
- `top_rules_summary.csv` - CSV summary of top rules
- `business_insights_report.txt` - Detailed business insights report

## 🔧 Configuration

### Adjusting Thresholds

Modify the thresholds in `main.py` or pass custom parameters:

```python
main(
    min_support=0.01,      # Lower for more rules, higher for fewer rules
    min_confidence=0.3,    # Lower for more rules, higher for stronger rules
    min_lift=1.0,          # Higher for stronger associations only
    max_items=100          # Increase for more items, decrease for faster processing
)
```

### Performance Considerations

- Lower `max_items` (50-100) for faster processing on large datasets
- Higher `min_support` reduces computation time significantly
- Use sample data for quick testing before running on full dataset

## 🐛 Troubleshooting

### No Rules Generated

If no rules are generated:
- Lower the `min_support` threshold
- Lower the `min_confidence` threshold
- Increase `max_items` to include more products
- Check if the dataset has sufficient transactions

### Kaggle API Authentication Failed

- Ensure `kaggle.json` is in `~/.kaggle/` directory
- Verify your Kaggle API token is valid
- Use sample data for testing: `main(use_sample_data=True)`

### Memory Issues

- Reduce `max_items` parameter
- Increase `min_support` threshold
- Filter by country to reduce dataset size

## 📚 References

- [Online Retail Dataset](https://www.kaggle.com/carrie1/online-retail)
- [Apriori Algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm)
- [Association Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning)
- [mlxtend Documentation](http://rasbt.github.io/mlxtend/)

## 📄 License

This project is for educational purposes. The dataset is from Kaggle and follows its respective license.

## 👤 Author

Created as a Principal Data Engineering project demonstrating end-to-end data pipeline development, from data ingestion to actionable business insights.

## 🤝 Contributing

This is a demonstration project. Feel free to fork and modify for your own use cases.

---

**Note:** For production use, consider implementing additional features such as:
- Real-time processing with streaming data
- Database integration for storing results
- Automated reporting and alerting
- Advanced algorithms like FP-Growth for better performance
- Time-based analysis for seasonal patterns
