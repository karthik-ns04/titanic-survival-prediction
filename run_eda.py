# run_eda.py - Complete Titanic EDA Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("="*80)
print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load data
print("\n📥 Loading data...")
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

# Display basic information
print(f"\n📊 DATASET SHAPES:")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

print(f"\n📋 TRAINING DATA COLUMNS:")
for i, col in enumerate(train_df.columns, 1):
    print(f"{i:2}. {col}")

print(f"\n🎯 TARGET VARIABLE: Survived")
print(f"   Survived: {train_df['Survived'].sum()} passengers ({train_df['Survived'].mean()*100:.1f}%)")
print(f"   Not Survived: {(train_df['Survived'] == 0).sum()} passengers ({100-train_df['Survived'].mean()*100:.1f}%)")

# Missing values analysis
print(f"\n🔍 MISSING VALUES ANALYSIS:")
print("Training data:")
missing_train = train_df.isnull().sum()
for col in train_df.columns:
    if missing_train[col] > 0:
        percent = (missing_train[col] / len(train_df)) * 100
        print(f"  {col}: {missing_train[col]} ({percent:.1f}%)")

print("\nTest data:")
missing_test = test_df.isnull().sum()
for col in test_df.columns:
    if missing_test[col] > 0:
        percent = (missing_test[col] / len(test_df)) * 100
        print(f"  {col}: {missing_test[col]} ({percent:.1f}%)")

# Survival rate visualization
print(f"\n📊 Creating survival visualization...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
survival_counts = train_df['Survived'].value_counts()
axes[0].pie(survival_counts, labels=['Not Survived', 'Survived'], 
           autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'])
axes[0].set_title('Survival Distribution')

# Bar chart
survival_counts.plot(kind='bar', ax=axes[1], color=['#ff6b6b', '#51cf66'])
axes[1].set_title('Survival Count')
axes[1].set_xlabel('Survived (0=No, 1=Yes)')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Not Survived', 'Survived'], rotation=0)

plt.tight_layout()

# Create reports folder if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Save the figure
plt.savefig('reports/figures/survival_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: reports/figures/survival_analysis.png")

# Show the plot
plt.show()

# Basic statistics
print(f"\n📈 BASIC STATISTICS:")
print("\nNumerical columns:")
print(train_df.describe())

print("\nCategorical columns:")
categorical_cols = train_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(train_df[col].value_counts().head())

# Additional analysis
print(f"\n📊 ADDITIONAL ANALYSIS:")

# Survival by Passenger Class
print("\n1. Survival by Passenger Class:")
class_survival = train_df.groupby('Pclass')['Survived'].mean() * 100
for pclass, rate in class_survival.items():
    print(f"   Class {pclass}: {rate:.1f}% survived")

# Survival by Gender
print("\n2. Survival by Gender:")
gender_survival = train_df.groupby('Sex')['Survived'].mean() * 100
for gender, rate in gender_survival.items():
    print(f"   {gender}: {rate:.1f}% survived")

# Age analysis
print("\n3. Age Statistics:")
print(f"   Average age: {train_df['Age'].mean():.1f} years")
print(f"   Youngest: {train_df['Age'].min():.0f} years")
print(f"   Oldest: {train_df['Age'].max():.0f} years")

print("\n" + "="*80)
print("🎉 EDA COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n📋 KEY FINDINGS:")
print("1. Dataset has 891 training samples and 418 test samples")
print("2. Survival rate: 38.4% survived, 61.6% did not survive")
print("3. Missing values in Age (~20%), Cabin (~77%), Embarked (0.2%)")
print("4. First class passengers had highest survival rate")
print("5. Women had much higher survival rate than men")
print("\n📁 Files created:")
print("   - reports/figures/survival_analysis.png (visualization)")
print("\n✅ Ready for Step 3: Data Preprocessing!")