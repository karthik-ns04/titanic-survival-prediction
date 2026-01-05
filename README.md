# 🚢 Titanic Survival Prediction

A complete machine learning project to predict Titanic passenger survival with interactive web interface.

## 📋 Project Overview

This project implements a machine learning pipeline to predict whether a passenger survived the Titanic disaster based on features like age, gender, class, and fare. The solution includes data preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit web application with filtering capabilities.

## 🎯 Features

- **Data Preprocessing**: Handles missing values, creates new features (Title, FamilySize, IsAlone, etc.)
- **Multiple ML Models**: Random Forest, Logistic Regression, Gradient Boosting, SVM
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Interactive Web App**: Streamlit interface with 4 modes:
  - 📊 Data Explorer: Explore Titanic dataset
  - 🤖 Make Predictions: Single & batch predictions
  - 📈 Model Insights: View model performance
  - ⚙️ Custom Filters: Filter & sort predictions
- **Filtering & Sorting**: Filter by class, gender, age, fare, survival status
- **Real-time Predictions**: Make predictions with probability scores

## 📊 Results

- **Best Model**: SVM (Support Vector Machine)
- **Validation Accuracy**: 82.68%
- **ROC-AUC Score**: 0.858
- **Precision**: 0.779
- **Recall**: 0.768

## 🛠️ Tech Stack

- **Python 3.9+**
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Version Control**: Git, GitHub

## 📁 Project Structure
titanic-survival-prediction/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── run_eda.py # EDA script
├── data/
│ └── raw/
│ ├── train.csv # Training data
│ └── test.csv # Test data
├── src/
│ ├── data/
│ │ └── preprocessing.py # Data preprocessing pipeline
│ ├── models/
│ │ └── train_model.py # Model training and evaluation
│ └── utils/
│ └── helpers.py # Utility functions
├── reports/
│ └── figures/ # Generated visualizations
├── models/ # Saved ML models
└── notebooks/ # Jupyter notebooks (optional)

text

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
2. Install dependencies
bash
pip install -r requirements.txt
3. Run EDA (Exploratory Data Analysis)
bash
python run_eda.py
4. Train the model
bash
python src/models/train_model.py
5. Launch the web app
bash
streamlit run app.py
📈 Model Performance
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
SVM	0.8268	0.7794	0.7681	0.7737	0.8578
Logistic Regression	0.8101	0.7692	0.7246	0.7463	0.8622
Random Forest	0.8045	0.7656	0.7101	0.7368	0.8390
Gradient Boosting	0.7765	0.7302	0.6667	0.6970	0.8306
🎥 Video Demonstration
[Link to video demonstration on YouTube/Vimeo]

🔧 Customization
Filtering Options in the App:
Filter by passenger class (1st, 2nd, 3rd)

Filter by gender (male/female)

Filter by age range

Filter by fare range

Filter by predicted survival status

Sort by survival probability, age, or fare

Adding New Features:
Edit src/data/preprocessing.py to add new feature engineering

Retrain model: python src/models/train_model.py

The Streamlit app will automatically use the updated model

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Kaggle for the Titanic dataset

Scikit-learn for ML algorithms

Streamlit for the web framework
