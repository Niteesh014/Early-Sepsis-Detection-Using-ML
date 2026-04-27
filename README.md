# Early Sepsis Detection Using Machine Learning

## Project Overview

This project uses the XGBoost algorithm to predict early sepsis risk using ICU clinical data. Early detection of sepsis can help accelerate intervention and improve patient outcomes.

## Objective

To build a classification model that predicts whether a patient is at risk of sepsis based on clinical features.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Google Colab

## Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Train-Test Split
6. XGBoost Model Training
7. Performance Evaluation

## Model Used

* XGBoost Classifier

## Results

The model was trained and evaluated on historical ICU patient data to identify sepsis risk patterns.

## Repository Files

* ESD_Using_ML.ipynb : Main notebook with preprocessing, training, and evaluation
* README.md : Project documentation

## Future Improvements

* Hyperparameter tuning
* Model explainability using SHAP
* External dataset validation
* Web deployment using Streamlit

## Disclaimer

This project is for academic and research purposes only and is not intended for clinical decision-making.

## Results

* Validation AUC: 0.8953
* Test AUC: 0.8911
* Optimized Classification Threshold: 0.2956
* Features Used: 56
* Model: XGBoost Classifier

The model demonstrated strong predictive capability for early sepsis risk detection on held-out test data.

