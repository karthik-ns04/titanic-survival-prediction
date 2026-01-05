# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """Complete Titanic data preprocessing pipeline"""
    
    def __init__(self):
        self.age_median = None
        self.fare_median = None
        self.embarked_mode = None
        
    def fit(self, X, y=None):
        # Calculate median and mode values for imputation
        self.age_median = X['Age'].median()
        self.fare_median = X['Fare'].median()
        self.embarked_mode = X['Embarked'].mode()[0] if len(X['Embarked'].mode()) > 0 else 'S'
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # 1. Handle missing values
        X['Age'] = X['Age'].fillna(self.age_median)
        X['Fare'] = X['Fare'].fillna(self.fare_median)
        X['Embarked'] = X['Embarked'].fillna(self.embarked_mode)
        X['Cabin'] = X['Cabin'].fillna('Unknown')
        
        # 2. Extract Title from Name
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Rare', 'Sir': 'Rare',
            'Mme': 'Mrs', 'Capt': 'Rare', 'Countess': 'Rare', 'Don': 'Rare',
            'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        X['Title'] = X['Title'].map(title_mapping).fillna('Rare')
        
        # 3. Create Family Features
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # 4. Extract Deck from Cabin
        X['Deck'] = X['Cabin'].str[0].fillna('U')
        
        # 5. Create Fare per Person - FIXED CODE
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']
        # Handle infinite values and NaN
        X['FarePerPerson'] = X['FarePerPerson'].replace([np.inf, -np.inf], np.nan)
        X['FarePerPerson'] = X['FarePerPerson'].fillna(X['Fare'])
        
        # 6. Age Groups
        X['AgeGroup'] = pd.cut(X['Age'], 
                              bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # 7. Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        for col in columns_to_drop:
            if col in X.columns:
                X = X.drop(columns=[col], errors='ignore')
        
        return X

def test_preprocessor():
    """Test the preprocessor"""
    print("Testing TitanicPreprocessor...")
    
    # Get the correct path
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'data', 'raw', 'train.csv')
    
    print(f"Looking for data at: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found at {data_path}")
        print("Make sure your CSV files are in data/raw/ folder")
        return None
    
    # Load sample data
    train_df = pd.read_csv(data_path)
    
    print(f"✅ Data loaded. Shape: {train_df.shape}")
    
    # Create and fit preprocessor
    preprocessor = TitanicPreprocessor()
    preprocessor.fit(train_df)
    
    # Transform data
    processed_data = preprocessor.transform(train_df)
    
    print(f"✅ Original shape: {train_df.shape}")
    print(f"✅ Processed shape: {processed_data.shape}")
    print(f"✅ Processed columns: {list(processed_data.columns)}")
    
    # Check for missing values
    print(f"\n✅ Missing values after processing:")
    missing = processed_data.isnull().sum().sum()
    if missing == 0:
        print(f"  No missing values! ✓")
    else:
        print(f"  Total missing values: {missing}")
        missing_cols = processed_data.isnull().sum()
        for col, count in missing_cols.items():
            if count > 0:
                print(f"  {col}: {count}")
    
    print("\n✅ First 3 rows of processed data:")
    print(processed_data.head(3))
    
    print("\n✅ Data types:")
    print(processed_data.dtypes)
    
    return processed_data

if __name__ == "__main__":
    test_preprocessor()