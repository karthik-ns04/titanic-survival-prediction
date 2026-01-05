# app.py - Titanic Survival Prediction Web App
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .survived {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .not-survived {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Find the latest model
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
                model_path = os.path.join(models_dir, latest_model)
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                st.sidebar.success(f"✅ Model loaded: {latest_model}")
                return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    return None

@st.cache_data
def load_original_data():
    """Load original Titanic data"""
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    return train_df, test_df

def predict_single_passenger(model_data, passenger_data):
    """Predict survival for a single passenger"""
    try:
        # Preprocess the passenger data
        preprocessor = model_data['preprocessor']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        model = model_data['model']
        
        # Convert to DataFrame
        passenger_df = pd.DataFrame([passenger_data])
        
        # Apply preprocessing
        processed_passenger = preprocessor.transform(passenger_df)
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in processed_passenger.columns:
                processed_passenger[col] = le.transform(processed_passenger[col])
        
        # Scale features
        scaled_features = scaler.transform(processed_passenger)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] if hasattr(model, 'predict_proba') else None
        
        return {
            'survived': bool(prediction),
            'probability': probability,
            'prediction': prediction
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg", 
                caption="RMS Titanic", use_column_width=True)
        
        st.markdown("---")
        st.markdown("### Navigation")
        app_mode = st.radio(
            "Select Mode:",
            ["📊 Data Explorer", "🤖 Make Predictions", "📈 Model Insights", "⚙️ Custom Filters"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This app predicts Titanic passenger survival using machine learning.
        
        **Features:**
        - Explore original Titanic data
        - Make individual predictions
        - Filter and sort predictions
        - View model performance
        """)
        
        # Load model
        model_data = load_model()
    
    # Main content based on selected mode
    if app_mode == "📊 Data Explorer":
        show_data_explorer()
    
    elif app_mode == "🤖 Make Predictions":
        if model_data:
            show_predictions(model_data)
        else:
            st.error("❌ No trained model found. Please train a model first.")
    
    elif app_mode == "📈 Model Insights":
        if model_data:
            show_model_insights(model_data)
        else:
            st.error("❌ No trained model found. Please train a model first.")
    
    elif app_mode == "⚙️ Custom Filters":
        if model_data:
            show_custom_filters(model_data)
        else:
            st.error("❌ No trained model found. Please train a model first.")

def show_data_explorer():
    """Show data exploration section"""
    st.markdown('<h2 class="sub-header">📊 Titanic Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # Load data
    train_df, test_df = load_original_data()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "🔍 Data Analysis", "📈 Visualizations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Data")
            st.dataframe(train_df.head(10), use_container_width=True)
            st.metric("Total Passengers", len(train_df))
            st.metric("Survival Rate", f"{(train_df['Survived'].mean() * 100):.1f}%")
        
        with col2:
            st.subheader("Test Data")
            st.dataframe(test_df.head(10), use_container_width=True)
            st.metric("Total Passengers", len(test_df))
            st.metric("Features", len(test_df.columns))
    
    with tab2:
        st.subheader("Data Statistics")
        
        # Select column to analyze
        selected_col = st.selectbox("Select column to analyze:", train_df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pd.api.types.is_numeric_dtype(train_df[selected_col]):
                st.write("**Statistics:**")
                st.write(train_df[selected_col].describe())
            else:
                st.write("**Value Counts:**")
                st.write(train_df[selected_col].value_counts())
        
        with col2:
            if selected_col == 'Survived':
                fig, ax = plt.subplots(figsize=(8, 6))
                survival_counts = train_df['Survived'].value_counts()
                colors = ['#ff6b6b', '#51cf66']
                ax.pie(survival_counts, labels=['Not Survived', 'Survived'], 
                      autopct='%1.1f%%', colors=colors)
                ax.set_title('Survival Distribution')
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Survival Visualizations")
        
        # Select visualization type
        viz_type = st.selectbox("Select visualization:", 
                               ["Survival by Class", "Survival by Gender", "Age Distribution", "Fare Distribution"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "Survival by Class":
            survival_by_class = train_df.groupby('Pclass')['Survived'].mean() * 100
            bars = ax.bar(survival_by_class.index.astype(str), survival_by_class.values, 
                         color=['#FFD700', '#C0C0C0', '#CD7F32'])
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Survival Rate (%)')
            ax.set_title('Survival Rate by Passenger Class')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
        
        elif viz_type == "Survival by Gender":
            survival_by_gender = train_df.groupby('Sex')['Survived'].mean() * 100
            colors = ['#1f77b4', '#ff7f0e']
            bars = ax.bar(survival_by_gender.index, survival_by_gender.values, color=colors)
            ax.set_xlabel('Gender')
            ax.set_ylabel('Survival Rate (%)')
            ax.set_title('Survival Rate by Gender')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
        
        elif viz_type == "Age Distribution":
            survived_age = train_df[train_df['Survived'] == 1]['Age'].dropna()
            not_survived_age = train_df[train_df['Survived'] == 0]['Age'].dropna()
            
            ax.hist([survived_age, not_survived_age], bins=20, 
                   label=['Survived', 'Not Survived'], 
                   color=['#51cf66', '#ff6b6b'], alpha=0.7)
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            ax.set_title('Age Distribution by Survival')
            ax.legend()
        
        elif viz_type == "Fare Distribution":
            survived_fare = train_df[train_df['Survived'] == 1]['Fare']
            not_survived_fare = train_df[train_df['Survived'] == 0]['Fare']
            
            ax.boxplot([survived_fare, not_survived_fare], 
                      labels=['Survived', 'Not Survived'])
            ax.set_ylabel('Fare')
            ax.set_title('Fare Distribution by Survival')
        
        st.pyplot(fig)

def show_predictions(model_data):
    """Show prediction interface"""
    st.markdown('<h2 class="sub-header">🤖 Make Predictions</h2>', unsafe_allow_html=True)
    
    # Two prediction modes
    prediction_mode = st.radio("Prediction Mode:", ["Single Passenger", "Batch Prediction"], horizontal=True)
    
    if prediction_mode == "Single Passenger":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Passenger Details")
            
            # Create form for passenger details
            with st.form("passenger_form"):
                pclass = st.selectbox("Passenger Class:", [1, 2, 3], index=2)
                sex = st.selectbox("Gender:", ["male", "female"])
                age = st.slider("Age:", 0.0, 100.0, 30.0, 0.5)
                sibsp = st.number_input("Siblings/Spouses:", 0, 10, 0)
                parch = st.number_input("Parents/Children:", 0, 10, 0)
                fare = st.number_input("Fare ($):", 0.0, 600.0, 32.2, 0.1)
                embarked = st.selectbox("Embarked:", ["C", "Q", "S"])
                
                submitted = st.form_submit_button("Predict Survival", type="primary")
        
        with col2:
            st.subheader("Prediction Result")
            
            if submitted:
                # Create passenger data
                passenger_data = {
                    'Pclass': pclass,
                    'Sex': sex,
                    'Age': age,
                    'SibSp': sibsp,
                    'Parch': parch,
                    'Fare': fare,
                    'Embarked': embarked,
                    'Name': 'Test Passenger',
                    'Ticket': 'TEST123',
                    'Cabin': 'Unknown',
                    'PassengerId': 999
                }
                
                # Make prediction
                result = predict_single_passenger(model_data, passenger_data)
                
                if result:
                    # Display result
                    if result['survived']:
                        st.markdown('<div class="prediction-box survived">', unsafe_allow_html=True)
                        st.markdown("## ✅ SURVIVED")
                        if result['probability']:
                            st.markdown(f"**Probability:** {result['probability']*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.balloons()
                    else:
                        st.markdown('<div class="prediction-box not-survived">', unsafe_allow_html=True)
                        st.markdown("## ❌ DID NOT SURVIVE")
                        if result['probability']:
                            st.markdown(f"**Probability:** {result['probability']*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show passenger details
                    st.markdown("### Passenger Summary:")
                    summary_df = pd.DataFrame({
                        'Feature': list(passenger_data.keys())[:7],
                        'Value': list(passenger_data.values())[:7]
                    })
                    st.table(summary_df)
            
            else:
                st.info("👆 Fill in passenger details and click 'Predict Survival'")
    
    else:  # Batch Prediction
        st.subheader("Batch Prediction")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with passenger data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_df = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data:**")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("Predict All Passengers", type="primary"):
                    # Process batch predictions
                    predictions = []
                    probabilities = []
                    
                    with st.spinner("Making predictions..."):
                        for _, row in batch_df.iterrows():
                            result = predict_single_passenger(model_data, row.to_dict())
                            if result:
                                predictions.append(result['survived'])
                                probabilities.append(result['probability'])
                            else:
                                predictions.append(None)
                                probabilities.append(None)
                    
                    # Add predictions to DataFrame
                    batch_df['Predicted_Survival'] = predictions
                    batch_df['Survival_Probability'] = probabilities
                    
                    # Display results
                    st.success(f"✅ Predictions complete for {len(batch_df)} passengers")
                    
                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        survived_count = sum(p for p in predictions if p)
                        st.metric("Predicted Survivors", survived_count)
                    with col2:
                        total_count = len([p for p in predictions if p is not None])
                        st.metric("Total Predicted", total_count)
                    with col3:
                        survival_rate = (survived_count / total_count * 100) if total_count > 0 else 0
                        st.metric("Predicted Survival Rate", f"{survival_rate:.1f}%")
                    
                    # Show results table
                    st.subheader("Prediction Results")
                    st.dataframe(batch_df[['PassengerId', 'Pclass', 'Sex', 'Age', 'Predicted_Survival', 'Survival_Probability']].head(20), 
                               use_container_width=True)
                    
                    # Download button
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="titanic_predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")

def show_model_insights(model_data):
    """Show model performance and insights"""
    st.markdown('<h2 class="sub-header">📈 Model Insights</h2>', unsafe_allow_html=True)
    
    # Display model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Best Model", model_data.get('best_model_name', 'Unknown'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        if 'results' in model_data and model_data['best_model_name'] in model_data['results']:
            accuracy = model_data['results'][model_data['best_model_name']]['accuracy']
            st.metric("Validation Accuracy", f"{accuracy*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Features", len(model_data.get('feature_names', [])))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show model comparison if available
    if 'results' in model_data:
        st.subheader("Model Performance Comparison")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(model_data['results']).T
        results_display = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4)
        
        # Display table
        st.dataframe(results_display, use_container_width=True)
        
        # Show visualizations if they exist
        st.subheader("Model Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Model comparison chart
            model_files = [
                'reports/figures/model_comparison.png',
                'reports/figures/roc_curves.png'
            ]
            
            for viz_file in model_files:
                if os.path.exists(viz_file):
                    st.image(viz_file, caption=os.path.basename(viz_file))
        
        with viz_col2:
            # Other visualizations
            viz_files = [
                'reports/figures/confusion_matrix.png',
                'reports/figures/feature_importance.png'
            ]
            
            for viz_file in viz_files:
                if os.path.exists(viz_file):
                    st.image(viz_file, caption=os.path.basename(viz_file))
    
    # Feature importance
    if 'feature_names' in model_data and model_data['feature_names']:
        st.subheader("Feature Importance")
        
        # For tree-based models
        if hasattr(model_data['model'], 'feature_importances_'):
            importance = model_data['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(feature_importance_df.head(10), use_container_width=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance_df.head(10)
            bars = ax.barh(top_features['Feature'], top_features['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            ax.invert_yaxis()
            st.pyplot(fig)

def show_custom_filters(model_data):
    """Show filtering and sorting interface"""
    st.markdown('<h2 class="sub-header">⚙️ Custom Filtering & Sorting</h2>', unsafe_allow_html=True)
    
    # Load test data
    test_df = pd.read_csv('data/raw/test.csv')
    
    # Make predictions for all test data
    st.info("Making predictions for all test passengers...")
    
    predictions = []
    probabilities = []
    
    with st.spinner("Processing predictions..."):
        for _, row in test_df.iterrows():
            result = predict_single_passenger(model_data, row.to_dict())
            if result:
                predictions.append(result['survived'])
                probabilities.append(result['probability'])
            else:
                predictions.append(None)
                probabilities.append(None)
    
    # Add predictions to DataFrame
    results_df = test_df.copy()
    results_df['Predicted_Survival'] = predictions
    results_df['Survival_Probability'] = probabilities
    results_df['Survival_Status'] = results_df['Predicted_Survival'].map({True: 'Survived', False: 'Not Survived'})
    
    st.success(f"✅ Predictions ready for {len(results_df)} passengers")
    
    # Filtering options
    st.subheader("Filter Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_class = st.multiselect("Passenger Class:", [1, 2, 3], default=[1, 2, 3])
    
    with col2:
        filter_gender = st.multiselect("Gender:", ["male", "female"], default=["male", "female"])
    
    with col3:
        filter_survival = st.multiselect("Predicted Survival:", ["Survived", "Not Survived"], 
                                        default=["Survived", "Not Survived"])
    
    # Age range filter
    age_min, age_max = st.slider("Age Range:", 
                                 float(results_df['Age'].min()), 
                                 float(results_df['Age'].max()), 
                                 (0.0, 80.0))
    
    # Fare range filter
    fare_min, fare_max = st.slider("Fare Range ($):", 
                                   float(results_df['Fare'].min()), 
                                   float(results_df['Fare'].max()), 
                                   (0.0, 100.0))
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if filter_class:
        filtered_df = filtered_df[filtered_df['Pclass'].isin(filter_class)]
    if filter_gender:
        filtered_df = filtered_df[filtered_df['Sex'].isin(filter_gender)]
    if filter_survival:
        survival_map = {'Survived': True, 'Not Survived': False}
        filtered_df = filtered_df[filtered_df['Predicted_Survival'].isin([survival_map[s] for s in filter_survival])]
    
    filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]
    filtered_df = filtered_df[(filtered_df['Fare'] >= fare_min) & (filtered_df['Fare'] <= fare_max)]
    
    # Sorting options
    st.subheader("Sort Results")
    
    sort_col1, sort_col2 = st.columns(2)
    
    with sort_col1:
        sort_by = st.selectbox("Sort by:", 
                              ['Survival_Probability', 'Age', 'Fare', 'Pclass', 'PassengerId'])
    
    with sort_col2:
        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
    
    # Apply sorting
    ascending = (sort_order == "Ascending")
    if sort_by == 'Survival_Probability':
        filtered_df = filtered_df.sort_values('Survival_Probability', ascending=ascending)
    else:
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Display results
    st.subheader(f"Filtered Results: {len(filtered_df)} passengers")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Filtered", len(filtered_df))
    with col2:
        survived_count = filtered_df['Predicted_Survival'].sum()
        st.metric("Predicted Survivors", int(survived_count))
    with col3:
        survival_rate = (survived_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    with col4:
        avg_fare = filtered_df['Fare'].mean()
        st.metric("Average Fare", f"${avg_fare:.2f}")
    
    # Display filtered data
    display_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 
                   'Predicted_Survival', 'Survival_Probability', 'Embarked']
    
    st.dataframe(filtered_df[display_cols].head(50), use_container_width=True)
    
    # Download filtered results
    if len(filtered_df) > 0:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Filtered Results ({len(filtered_df)} rows)",
            data=csv,
            file_name="filtered_titanic_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()