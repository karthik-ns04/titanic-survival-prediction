# src/models/train_model.py
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import custom modules
try:
    from src.data.preprocessing import TitanicPreprocessor
    print("✅ Successfully imported TitanicPreprocessor")
except ImportError as e:
    print(f"⚠️  Could not import TitanicPreprocessor: {e}")
    print("⚠️  Using simplified preprocessing instead")
    
    # Define a simple preprocessor if import fails
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class TitanicPreprocessor(BaseEstimator, TransformerMixin):
        """Simple Titanic data preprocessing pipeline"""
        
        def __init__(self):
            self.age_median = None
            self.fare_median = None
            self.embarked_mode = None
            
        def fit(self, X, y=None):
            self.age_median = X['Age'].median()
            self.fare_median = X['Fare'].median()
            self.embarked_mode = X['Embarked'].mode()[0] if len(X['Embarked'].mode()) > 0 else 'S'
            return self
        
        def transform(self, X, y=None):
            X = X.copy()
            X['Age'] = X['Age'].fillna(self.age_median)
            X['Fare'] = X['Fare'].fillna(self.fare_median)
            X['Embarked'] = X['Embarked'].fillna(self.embarked_mode)
            X['Cabin'] = X['Cabin'].fillna('Unknown')
            
            # Extract Title
            X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
            X['Title'] = X['Title'].map(title_mapping).fillna('Rare')
            
            # Family features
            X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
            X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
            
            # Drop columns
            columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
            for col in columns_to_drop:
                if col in X.columns:
                    X = X.drop(columns=[col], errors='ignore')
            
            return X

class TitanicModelTrainer:
    """Train and evaluate machine learning models for Titanic survival prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = TitanicPreprocessor()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_data(self, train_path='../data/raw/train.csv', test_path='../data/raw/test.csv'):
        """Load and prepare data for training"""
        print("="*60)
        print("PREPARING DATA FOR MODEL TRAINING")
        print("="*60)
        
        # Load data with absolute paths
        train_path = os.path.join(project_root, 'data', 'raw', 'train.csv')
        test_path = os.path.join(project_root, 'data', 'raw', 'test.csv')
        
        print(f"Looking for training data at: {train_path}")
        print(f"Looking for test data at: {test_path}")
        
        if not os.path.exists(train_path):
            print(f"❌ ERROR: Training file not found at {train_path}")
            print("Please make sure your CSV files are in data/raw/ folder")
            return None
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        
        print(f"✅ Training data shape: {train_df.shape}")
        if test_df is not None:
            print(f"✅ Test data shape: {test_df.shape}")
        
        # Fit preprocessor on training data
        print("\n🔄 Preprocessing data...")
        self.preprocessor.fit(train_df)
        
        # Transform both datasets
        X_train_processed = self.preprocessor.transform(train_df)
        X_test_processed = self.preprocessor.transform(test_df) if test_df is not None else None
        
        # Separate features and target for training
        y_train = X_train_processed['Survived']
        X_train = X_train_processed.drop('Survived', axis=1)
        
        print(f"✅ Processed X_train shape: {X_train.shape}")
        print(f"✅ Processed y_train shape: {y_train.shape}")
        
        # Encode categorical variables
        if X_test_processed is not None:
            X_train_encoded, X_test_encoded = self._encode_categorical(X_train, X_test_processed)
        else:
            X_train_encoded, _ = self._encode_categorical(X_train, X_train)
            X_test_encoded = None
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_encoded, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val_split)
        X_test_scaled = self.scaler.transform(X_test_encoded) if X_test_encoded is not None else None
        
        print(f"✅ Final shapes:")
        print(f"   X_train: {X_train_scaled.shape}")
        print(f"   X_val: {X_val_scaled.shape}")
        if X_test_scaled is not None:
            print(f"   X_test: {X_test_scaled.shape}")
        
        self.feature_names = X_train.columns.tolist()
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_split,
            'y_val': y_val_split,
            'feature_names': self.feature_names
        }
    
    def _encode_categorical(self, X_train, X_test):
        """Encode categorical variables"""
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in X_train_encoded.columns:
                le = LabelEncoder()
                # Combine train and test to ensure all categories are encoded
                combined = pd.concat([X_train_encoded[col], X_test_encoded[col]], axis=0)
                le.fit(combined)
                
                X_train_encoded[col] = le.transform(X_train_encoded[col])
                X_test_encoded[col] = le.transform(X_test_encoded[col])
                
                self.label_encoders[col] = le
        
        return X_train_encoded, X_test_encoded
    
    def train_models(self, data):
        """Train multiple machine learning models"""
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            )
        }
        
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            try:
                model.fit(data['X_train'], data['y_train'])
                self.models[name] = model
                
                # Make predictions
                y_pred = model.predict(data['X_val'])
                y_pred_proba = model.predict_proba(data['X_val'])[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(data['y_val'], y_pred)
                precision = precision_score(data['y_val'], y_pred, zero_division=0)
                recall = recall_score(data['y_val'], y_pred, zero_division=0)
                f1 = f1_score(data['y_val'], y_pred, zero_division=0)
                roc_auc = roc_auc_score(data['y_val'], y_pred_proba) if y_pred_proba is not None else 0
                
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': confusion_matrix(data['y_val'], y_pred)
                }
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   ✅ Precision: {precision:.4f}")
                print(f"   ✅ Recall: {recall:.4f}")
                print(f"   ✅ F1 Score: {f1:.4f}")
                print(f"   ✅ ROC AUC: {roc_auc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
        
        # Select best model based on accuracy
        if self.results:
            self.best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
            self.best_model = self.models[self.best_model_name]
            
            print(f"\n🏆 BEST MODEL: {self.best_model_name}")
            print(f"   Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        else:
            print("❌ No models were successfully trained")
            return None
        
        return self.results
    
    def evaluate_models(self, data):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        # Create comparison table
        results_df = pd.DataFrame(self.results).T
        print("\n📊 MODEL COMPARISON:")
        print(results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4))
        
        # Plot comparison
        self._plot_model_comparison()
        
        # Plot confusion matrix for best model
        self._plot_confusion_matrix(data)
        
        # Plot ROC curves
        self._plot_roc_curves(data)
        
        # Feature importance for tree-based models
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            self._plot_feature_importance(data['feature_names'])
        
        return results_df
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        if not self.results:
            print("⚠️  No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            model_names = list(self.results.keys())
            scores = [self.results[name][metric] for name in model_names]
            
            bars = ax.bar(model_names, scores, color=sns.color_palette('husl', len(model_names)))
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Create reports folder if it doesn't exist
        os.makedirs('reports/figures', exist_ok=True)
        
        # Save figure
        figure_path = os.path.join(project_root, 'reports', 'figures', 'model_comparison.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved: {figure_path}")
        plt.show()
    
    def _plot_confusion_matrix(self, data):
        """Plot confusion matrix for best model"""
        if not self.best_model:
            return
        
        y_pred = self.best_model.predict(data['X_val'])
        cm = confusion_matrix(data['y_val'], y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Survived', 'Survived'],
                   yticklabels=['Not Survived', 'Survived'])
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        figure_path = os.path.join(project_root, 'reports', 'figures', 'confusion_matrix.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {figure_path}")
        plt.show()
    
    def _plot_roc_curves(self, data):
        """Plot ROC curves for all models"""
        if not self.models:
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(data['X_val'])[:, 1]
                    fpr, tpr, _ = roc_curve(data['y_val'], y_pred_proba)
                    roc_auc = roc_auc_score(data['y_val'], y_pred_proba)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        figure_path = os.path.join(project_root, 'reports', 'figures', 'roc_curves.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved: {figure_path}")
        plt.show()
    
    def _plot_feature_importance(self, feature_names):
        """Plot feature importance for tree-based models"""
        if not self.best_model or not hasattr(self.best_model, 'feature_importances_'):
            return
        
        importance = self.best_model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = min(15, len(self.feature_importance))
        bars = plt.barh(self.feature_importance['feature'][:top_features], 
                       self.feature_importance['importance'][:top_features])
        plt.xlabel('Importance')
        plt.title(f'Top {top_features} Feature Importance - {self.best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center')
        
        plt.tight_layout()
        figure_path = os.path.join(project_root, 'reports', 'figures', 'feature_importance.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved: {figure_path}")
        plt.show()
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        print("\n" + "="*60)
        print("SAVING MODEL & ARTIFACTS")
        print("="*60)
        
        # Create models directory
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best model
        model_filename = f'titanic_model_{timestamp}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'preprocessor': self.preprocessor,
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'feature_names': self.feature_names,
                    'results': self.results,
                    'best_model_name': self.best_model_name
                }, f)
            
            print(f"✅ Model saved: {model_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
        
        return model_path

def main():
    """Main training pipeline"""
    print("🚀 STARTING TITANIC SURVIVAL PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = TitanicModelTrainer(random_state=42)
    
    # Prepare data
    data = trainer.prepare_data()
    if data is None:
        print("❌ Failed to prepare data. Exiting.")
        return None
    
    # Train models
    results = trainer.train_models(data)
    if results is None:
        print("❌ Failed to train models. Exiting.")
        return None
    
    # Evaluate models
    trainer.evaluate_models(data)
    
    # Save model
    model_path = trainer.save_model()
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    if model_path:
        print(f"\n📋 SUMMARY:")
        print(f"• Best Model: {trainer.best_model_name}")
        print(f"• Validation Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        print(f"• Models Trained: {len(trainer.models)}")
        print(f"• Model saved to: {model_path}")
        print(f"• Visualizations saved to: reports/figures/")
    
    return trainer

if __name__ == "__main__":
    trainer = main()