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

## Model Performance

* ROC-AUC: 0.8911
* Accuracy: 89.47%
* Recall (Sepsis Detection): 71.82%
* Precision: 38.11%
* F1 Score: 49.80%
* Threshold Used: 0.2956

The model was tuned to prioritize recall for early sepsis detection, which is important in clinical screening scenarios.

## Images

## ROC CURVE

<img width="790" height="590" alt="ROC CURVE" src="https://github.com/user-attachments/assets/0b74e4e2-742a-46c7-9047-adea3511610d" />

## Confusion Matrix

<img width="666" height="590" alt="CM" src="https://github.com/user-attachments/assets/ff6a987f-3ec5-4c8a-80d3-730fcc073aff" />

## Feature Importance

<img width="989" height="690" alt="FI" src="https://github.com/user-attachments/assets/b75fe92f-c603-4998-b117-bfcf3a79e184" />






