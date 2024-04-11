import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matminer.datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pymatgen.core.structure import Structure
import pickle

# Load matbench_dielectric dataset
data = load_dataset("matbench_dielectric")

# Extract features and targets
structures = data['structure']
y = data['n']

# Function to extract structural features
def get_structure_features(structure):
    return [structure.num_sites, structure.volume, structure.density]

# Extract structural features for all structures
X = pd.DataFrame([get_structure_features(structure) for structure in structures],
                 columns=["num_sites", "volume", "density"])

# Load the trained model
@st.cache_data()
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to predict dielectric constant
def predict_dielectric(model, composition):
    # Convert composition to features (num_sites, volume, density)
    # You need to implement this based on your data preprocessing
    # For example, if your model expects a Pandas DataFrame X_test as input, you can create it here
    X_test = pd.DataFrame([get_structure_features(structure) for structure in structures],
                          columns=["num_sites", "volume", "density"])
    
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

def obtain_y_test():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test

# Function to display metrics and plots
def display_metrics_and_plots(y_test, y_pred=None):
    if y_pred is not None:
        # Calculate Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        st.write("Mean Absolute Error:", mae)

        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.title('Predicted vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        st.pyplot()
    else:
        st.write("No predictions available.")
# Main function to run the Streamlit app
# Main function to run the Streamlit app
def main():
    # Load the model
    model = load_model()
    y_test = obtain_y_test()

    # Initialize y_pred
    y_pred = None

    # Title and description
    st.title('Dielectric Constant Prediction')
    st.write('This app predicts the dielectric constant using a machine learning model.')

    # Add a form for user input
    composition = st.text_input('Enter composition (e.g., CuO2):')
    if st.button('Predict'):
        # Perform prediction
        y_pred = predict_dielectric(model, composition)
        st.write('Predicted Dielectric Constant:', y_pred)

    # Optionally, display metrics and plots if y_pred is not None
    if y_pred is not None:
        display_metrics_and_plots(y_test, y_pred)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
