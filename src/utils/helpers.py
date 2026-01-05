# src/utils/helpers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_separator(text):
    """Print formatted separator"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_data_quality(df, name="Dataset"):
    """Check data quality and print report"""
    print_separator(f"DATA QUALITY CHECK - {name}")
    
    print(f"Shape: {df.shape}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            percent = (missing[col] / len(df)) * 100
            print(f"  {col}: {missing[col]} ({percent:.1f}%)")
    
    print("\nData types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    print("\nSample data:")
    print(df.head(3))

def save_visualization(fig, filename, dpi=300):
    """Save matplotlib figure"""
    os.makedirs('reports/figures', exist_ok=True)
    fig.savefig(f'reports/figures/{filename}', dpi=dpi, bbox_inches='tight')
    print(f"✅ Figure saved to reports/figures/{filename}")

def create_summary_report(train_df, test_df, output_file='reports/data_summary.txt'):
    """Create a summary report of the data"""
    os.makedirs('reports', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATA SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. DATASET SIZES\n")
        f.write(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns\n")
        f.write(f"Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns\n\n")
        
        f.write("2. SURVIVAL STATISTICS\n")
        survival_rate = (train_df['Survived'].sum() / len(train_df)) * 100
        f.write(f"Survival rate: {survival_rate:.2f}%\n")
        f.write(f"Survived: {train_df['Survived'].sum()}\n")
        f.write(f"Not Survived: {len(train_df) - train_df['Survived'].sum()}\n\n")
        
        f.write("3. KEY INSIGHTS\n")
        f.write("- Women had much higher survival rate than men\n")
        f.write("- First class passengers had highest survival rate\n")
        f.write("- Age and Fare show correlation with survival\n")
        
    print(f"✅ Summary report saved to {output_file}")