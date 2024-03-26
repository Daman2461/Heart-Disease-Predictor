# Heart-Disease-Predictor


## Overview

This Python script is designed to predict heart disease using machine learning models, specifically Random Forest and XGBoost classifiers. It provides users with the ability to train, evaluate, and make predictions based on input data.

## Key Features

- **Data Loading and Preprocessing:** The script loads the heart disease dataset (`heart.csv`) using pandas and preprocesses it by encoding categorical variables using one-hot encoding.

- **Model Training and Evaluation:** Users can choose to train and evaluate Random Forest and XGBoost models on the preprocessed data. Training metrics such as accuracy are displayed for both models.

- **Prediction:** The script allows users to input a single data point in the form of a 2D list and obtain predictions using either the trained Random Forest or XGBoost model.

- **User Interaction:** A menu-driven interface guides users through the available options, including training models, making predictions, and exiting the program.

## Usage Instructions

1. **Clone the repository to your local machine:**

   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   pip install pandas scikit-learn xgboost
   python heart_disease_prediction.py ```

## Dataset

   -The heart disease dataset (heart.csv) contains various features such as age, blood pressure, cholesterol levels, and categorical variables related to chest pain, ECG results, and exercise-induced angina.Dataset taken from Kaggle
## Example

![image](https://github.com/Daman2461/Heart-Disease-Predictor/assets/142651012/c49e3b6b-54cc-4fad-a756-0c03d3b321e7)
![image](https://github.com/Daman2461/Heart-Disease-Predictor/assets/142651012/ee63fe07-e5fa-4b3e-a31d-dedba1c6b5b4)
![image](https://github.com/Daman2461/Heart-Disease-Predictor/assets/142651012/b4ce2cd8-71e9-426c-a039-f1254ff1bb5c)

