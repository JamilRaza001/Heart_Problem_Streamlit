import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset and model
df = pd.read_csv('heart.csv')
df.drop_duplicates(inplace=True)
model = joblib.load('logistic_regression_model.pkl')

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

if section == "Dashboard":
    st.title("Heart Disease Dashboard")
    st.write("Explore visualizations of the heart disease dataset.")

    # Bar plot of target distribution
    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, ax=ax)
    ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Scatter plot of age vs cholesterol
    st.subheader("Age vs Cholesterol by Target")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='chol', hue='target', data=df, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol (mg/dl)")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    st.pyplot(fig)

    # Boxplot of resting blood pressure by target
    st.subheader("Resting Blood Pressure by Target")
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y='trestbps', data=df, ax=ax)
    ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
    ax.set_ylabel("Resting Blood Pressure (mmHg)")
    st.pyplot(fig)

    # Histogram of age
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], bins=20, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif section == "Prediction":
    st.title("Heart Disease Prediction")
    st.write("Enter patient data to predict the likelihood of heart disease.")

    # Feature inputs
    feature_names = df.columns.drop('target')
    inputs = {}
    st.header("Patient Data Input")

    for feature in feature_names:
        if feature in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            # Numerical features
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            inputs[feature] = st.number_input(f"{feature} (Range: {min_val} - {max_val})", 
                                             min_value=min_val, max_value=max_val, value=min_val)
        else:
            # Categorical features
            unique_vals = sorted(df[feature].unique())
            min_val = int(min(unique_vals))
            max_val = int(max(unique_vals))
            inputs[feature] = st.number_input(f"{feature} (Range: {min_val} - {max_val})", 
                                             min_value=min_val, max_value=max_val, value=min_val, step=1)

    # Prediction button
    if st.button("Predict"):
        try:
            input_list = [inputs[feature] for feature in feature_names]
            input_array = np.array(input_list).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][prediction]

            # Display result with improved clarity
            if prediction == 0:
                st.success(f"The model predicts that the patient does not have heart disease with a probability of {probability:.2f}.")
            else:
                st.error(f"The model predicts that the patient has heart disease with a probability of {probability:.2f}.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")