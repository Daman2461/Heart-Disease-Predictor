import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def display_menu():
    print("\n\nWelcome! Please choose an option:")
    print("1. Train and Evaluate Random Forest")
    print("2. Train and Evaluate XGBoost")
    print("3. Predict using XGBoost")
    print("4. Predict using Random Forest")
    print("5. Take Input for Prediction (Single Data Point)")
    print("6. Exit")

def train_evaluate_random_forest(X_train, X_val, y_train, y_val):
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=10, random_state=55)
    model_rf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model_rf.predict(X_train))
    val_accuracy = accuracy_score(y_val, model_rf.predict(X_val))
    print("\n\n\nRandom Forest Model Metrics:")
    print(f"\tTraining Accuracy: {train_accuracy:.4f}")
    print(f"\tValidation Accuracy: {val_accuracy:.4f}")
    return model_rf

def train_evaluate_xgboost(X_train, X_val, y_train, y_val):
    model_xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, verbosity=1, random_state=55)
    model_xgb.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model_xgb.predict(X_train))
    val_accuracy = accuracy_score(y_val, model_xgb.predict(X_val))
    print("\n\n\nXGBoost Model Metrics:")
    print(f"\tTraining Accuracy: {train_accuracy:.4f}")
    print(f"\tValidation Accuracy: {val_accuracy:.4f}")
    return model_xgb

def predict_single_data_point(model, features,pred):
    prediction = model.predict(pred)
    print(f"\nThe model predicts {'Heart Disease' if prediction == 1 else 'No Heart Disease'} for the given data point.")

# Load the dataset using pandas
df = pd.read_csv("heart.csv")

# Data preprocessing
categories_to_encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(data=df, prefix=categories_to_encode, columns=categories_to_encode)
features = [x for x in df.columns if x != 'HeartDisease']
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=55)

# Train and evaluate Random Forest and XGBoost
model_rf = None
model_xgb = None
pred=None
while True:
    display_menu()
    choice = input("Enter your choice (1/2/3/4/5/6): ")

    if choice == '1':
        model_rf = train_evaluate_random_forest(X_train, X_val, y_train, y_val)
    elif choice == '2':
        model_xgb = train_evaluate_xgboost(X_train, X_val, y_train, y_val)
    elif choice == '3':
        if model_xgb is None:
            print("Please train an XGBoost model first.")
        else:
            predict_single_data_point(model_xgb, features,pred)
    elif choice == '4':
        if model_rf is None:
            print("Please train a Random Forest model first.")
        else:
            predict_single_data_point(model_rf, features,pred)
    elif choice == '5':
        
            pred = eval(input("Enter the data point as a 2D-list of values separated by commas: "))
    elif choice == '6':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please choose a valid option.")
