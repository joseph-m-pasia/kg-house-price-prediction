## 🏠 House Prices Prediction (Machine Learning Project)

A machine learning pipeline for predicting house prices using structured housing data.
This project demonstrates a production-style ML workflow including data preprocessing, feature engineering, model training, evaluation, and reproducible experimentation.

The project follows a clean src-based architecture commonly used in professional ML and data science projects.

## Project Goals

The objective is to build models that accurately predict house prices based on property attributes.
Key aspects of the project include:
 - Exploratory Data Analysis (EDA)
 - Data preprocessing
 - Feature engineering
 - Model training and evaluation
 - Reproducible ML pipelines
 - Clean project structure for scalability

## Machine Learning Pipeline

Raw Data -> Data Cleaning -> Feature Engineering -> Train/Test Split -> Model Training -> Save Model -> Model Evaluation -> Model Selection -> Champion Model Hyperparameter Optimization

## Installation

git clone https://github.com/yourusername/kg-house-price-prediction.git

cd kg-house-price-prediction

#### Install the package in editable mode:
pip install -e .

This allows Python to import the project package: pkg_house_prices

## Dependencies

Main libraries used in this project:
 - TensorFlow
 - Pandas
 - NumPy
 - Scikit-learn
 - Matplotlib
 - Dependencies are defined in pyproject.toml.

## Running the Training Pipeline

From the project root: 

python -m pkg_house_prices.models.task_01_train_models

This will:
 - Load the dataset
 - Process and transform features
 - Train machine learning models
 - Evaluate model performance
 - Save trained models
