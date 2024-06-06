import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to remove outliers using IQR
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
    data = pd.read_csv(url)
    data1 = pd.read_csv(url)
    return data, data1

# Exploratory Data Analysis
def exploratory_data_analysis(data):
    st.subheader("Exploratory Data Analysis")
    st.write(f'DataFrame size: {data.shape}')
    st.write(data.head())
    st.write(data.info())
    st.write(data.describe())

    # Plot histograms
    st.subheader("Histograms of Numerical Features")
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 15))
    for i, column in enumerate(numeric_columns):
        sns.histplot(data[column], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Median House Value vs Median Income")
    fig = px.scatter(data, x="median_income", y="median_house_value", trendline="ols",
                     color='median_house_value', color_continuous_scale=px.colors.sequential.Oranges)
    st.plotly_chart(fig)

    # Box plot
    st.subheader("Median House Value vs Ocean Proximity")
    fig = px.box(data, x="ocean_proximity", y="median_house_value", color='ocean_proximity')
    st.plotly_chart(fig)

# Data Preprocessing
def data_preprocessing(data):
    st.subheader("Data Preprocessing")

    # Handle missing values
    st.write("Handling missing values...")
    data.dropna(subset=['total_bedrooms'], inplace=True)

    # Feature Engineering
    st.write("Feature engineering...")
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']

    # Encoding categorical variables
    data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

    # Scaling numerical features
    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income', 'rooms_per_household',
                          'bedrooms_per_room', 'population_per_household']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Create interaction features
    data['income_times_rooms'] = data['median_income'] * data['total_rooms']

    # Remove outliers
    st.write("Removing outliers...")
    data_cleaned = remove_outliers(data, numerical_features + ['median_house_value'])
    st.write(f"Original data shape: {data.shape}")
    st.write(f"Cleaned data shape: {data_cleaned.shape}")

    return data, data_cleaned

# Model Training and Evaluation
def model_training(data):
    st.subheader("Model Training and Evaluation")

    # Use a subset of data for faster training during development
    data = data.sample(frac=0.3, random_state=42)

    # Feature Vector and Target Variable
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Training Random Forest Model...")
    with st.spinner('Training Random Forest...'):
        # Train Random Forest Model
        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

    st.write("Training MLP Regressor Model...")
    with st.spinner('Training MLP Regressor...'):
        # Train MLP Regressor Model
        model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                 max_iter=5000, random_state=42, learning_rate_init=0.001,
                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=100)
        model_mlp.fit(X_train, y_train)
        y_pred_mlp = model_mlp.predict(X_test)
        mse_mlp = mean_squared_error(y_test, y_pred_mlp)
        r2_mlp = r2_score(y_test, y_pred_mlp)

    # Display metrics
    st.write(f"Random Forest - Mean Squared Error: {mse_rf}, R^2 Score: {r2_rf}")
    st.write(f"MLP Regressor - Mean Squared Error: {mse_mlp}, R^2 Score: {r2_mlp}")

    # Feature Importances for Random Forest
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], align='center')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    ax.invert_yaxis()
    st.pyplot(fig)

    # Plot Prediction vs. Actual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred_rf, edgecolors=(0, 0, 0))
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # Plot Residuals
    residuals = y_test - y_pred_rf
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred_rf, residuals, edgecolors=(0, 0, 0))
    ax.hlines(0, min(y_pred_rf), max(y_pred_rf), colors='r', linestyles='dashed')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    st.pyplot(fig)

def main():
    st.title("California Housing Price Prediction")

    # Load Data
    data, data1 = load_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Exploratory Data Analysis", "Data Preprocessing", "Model Training and Evaluation"]
    choice = st.sidebar.radio("Choose an option", options)

    if choice == "Exploratory Data Analysis":
        exploratory_data_analysis(data)
    elif choice == "Data Preprocessing":
        actual_data, cleaned_data = data_preprocessing(data)
        st.write(cleaned_data.head())
        
        st.subheader("Download Data")
        st.download_button(
            label="Download Cleaned Dataset as CSV",
            data=cleaned_data.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_dataset.csv',
            mime='text/csv'
        )
        
        st.download_button(
            label="Download Original Dataset as CSV",
            data=data1.to_csv(index=False).encode('utf-8'),
            file_name='original_dataset.csv',
            mime='text/csv'
        )
    elif choice == "Model Training and Evaluation":
        actual_data, cleaned_data = data_preprocessing(data)
        model_training(cleaned_data)

if __name__ == '__main__':
    main()
