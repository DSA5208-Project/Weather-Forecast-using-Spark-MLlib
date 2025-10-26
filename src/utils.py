"""
Utility functions for data exploration and analysis
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_dataset(df):
    """
    Perform basic exploratory data analysis
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"\nTotal rows: {df.count()}")
    print(f"Total columns: {len(df.columns)}")
    
    # Schema
    print("\nSchema:")
    df.printSchema()
    
    # Sample data
    print("\nSample data:")
    df.show(5, truncate=True)
    
    # Column names
    print("\nAvailable columns:")
    for i, col_name in enumerate(df.columns, 1):
        print(f"  {i}. {col_name}")
    
    return df


def check_missing_values(df):
    """
    Check for missing values in DataFrame
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    total_rows = df.count()
    
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        null_percentage = (null_count / total_rows) * 100
        
        if null_percentage > 0:
            print(f"{column}: {null_count} ({null_percentage:.2f}%)")


def get_statistics(df, numeric_columns):
    """
    Get descriptive statistics for numeric columns
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
    """
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    df.select(numeric_columns).describe().show()


def plot_distributions(df, columns, output_path):
    """
    Plot distributions of selected columns
    
    Args:
        df: Input DataFrame
        columns: List of column names to plot
        output_path: Path to save the plot
    """
    # Sample data for plotting
    sample_df = df.select(columns).sample(fraction=0.01, seed=42).toPandas()
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, column in enumerate(columns):
        axes[idx].hist(sample_df[column].dropna(), bins=50, edgecolor='black')
        axes[idx].set_xlabel(column)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {column}')
        axes[idx].grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    from data_preprocessing import create_spark_session
    
    spark = create_spark_session()
    
    try:
        # Load sample data
        data_path = "data/2024/*.csv"
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Analyze
        analyze_dataset(df)
        check_missing_values(df)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        spark.stop()
