import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained LSTM model
@st.cache_resource
def load_lstm_model():
    model = load_model("cnc_lstm_model.h5")
    return model

# Preprocess the uploaded CSV file
def preprocess_data(df, expected_features):
    """
    Preprocess the uploaded CSV file to match the expected number of features.
    """
    # Drop unnecessary columns
    columns_to_drop = ['No', 'material']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Handle missing values for numeric columns only
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Label encode 'Machining_Process' column
    if 'Machining_Process' in df.columns:
        # Convert to lowercase
        df['Machining_Process'] = df['Machining_Process'].str.lower()

        # Group similar stages
        process_mapping = {
            'layer 1 up': 'layer up', 'layer 2 up': 'layer up', 'layer 3 up': 'layer up',
            'layer 1 down': 'layer down', 'layer 2 down': 'layer down', 'layer 3 down': 'layer down',
            'end': 'end'
        }
        df['Machining_Process'] = df['Machining_Process'].replace(process_mapping)

        # Label encode
        label_encoder = LabelEncoder()
        df['Machining_Process'] = label_encoder.fit_transform(df['Machining_Process'])
        

    # Normalize numerical features
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Ensure the number of features matches the expected number
    current_features = df.shape[1]
    if current_features < expected_features:
        # Pad with zeros if there are fewer features
        padding = pd.DataFrame(np.zeros((df.shape[0], expected_features - current_features)))
        df = pd.concat([df, padding], axis=1)
    elif current_features > expected_features:
        # Drop extra features if there are more
        df = df.iloc[:, :expected_features]

    return df

def make_predictions(model, data):
    """
    Make predictions using the LSTM model.
    """
    # Reshape data for LSTM input
    X_reshaped = data.values.reshape((data.shape[0], 1, data.shape[1]))
    predictions = model.predict(X_reshaped)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels
    return predictions

# Streamlit App
st.set_page_config(page_title="CNC Machine Prediction App", page_icon="⚙️", layout="wide")

# Title and description
st.title("⚙️ CNC Machine Prediction App")
st.markdown("""
    Upload a CNC experiment sensor CSV file, and the app will predict:
    - **Tool Condition** (Worn/Unworn)
    - **Machining Completion** (Yes/No)
    - **Passed Visual Inspection** (Yes/No)
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display the uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Load the LSTM model and get the expected number of features
    model = load_lstm_model()
    expected_features = model.input_shape[-1]  # Get the number of features expected by the model

    # Preprocess the data to match the expected number of features
    processed_data = preprocess_data(df, expected_features)
    
    # Debug: Display preprocessed data
    st.subheader("Preprocessed Data")
    st.write(processed_data.head())

    # Make predictions
    predictions = make_predictions(model, processed_data)
    
    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame(predictions, columns=[
        'Predicted_Tool_Condition', 
        'Predicted_Machining_Completed', 
        'Predicted_Passed_Visual_Inspection'
    ])
    
    # Map binary predictions to human-readable labels
    predictions_df['Predicted_Tool_Condition'] = predictions_df['Predicted_Tool_Condition'].map({0: 'Unworn', 1: 'Worn'})
    predictions_df['Predicted_Machining_Completed'] = predictions_df['Predicted_Machining_Completed'].map({0: 'No', 1: 'Yes'})
    predictions_df['Predicted_Passed_Visual_Inspection'] = predictions_df['Predicted_Passed_Visual_Inspection'].map({0: 'No', 1: 'Yes'})

    # Display predictions
    st.subheader("Predictions")
    st.write(predictions_df)

    # Visualizations
    st.subheader("Data Visualizations")

    # Plot 1: Tool Condition Distribution
    st.markdown("**Tool Condition Distribution**")
    fig1, ax1 = plt.subplots(figsize=(8, 4))  # Medium size
    sns.countplot(x='Predicted_Tool_Condition', data=predictions_df, palette='coolwarm', ax=ax1)
    st.pyplot(fig1)

    # Plot 2: Machining Completion Distribution
    st.markdown("**Machining Completion Distribution**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))  # Medium size
    sns.countplot(x='Predicted_Machining_Completed', data=predictions_df, palette='viridis', ax=ax2)
    st.pyplot(fig2)

    # Plot 3: Passed Visual Inspection Distribution
    st.markdown("**Passed Visual Inspection Distribution**")
    fig3, ax3 = plt.subplots(figsize=(8, 4))  # Medium size
    sns.countplot(x='Predicted_Passed_Visual_Inspection', data=predictions_df, palette='magma', ax=ax3)
    st.pyplot(fig3)


else:
    st.info("👆 Please upload a CSV file to get started.")