import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load Data
data = pd.read_csv(r"D:\Project Datasets\Liver_disease_data.csv")

# Split features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
keys = X.columns

# Scaling
scale = MinMaxScaler()
X = scale.fit_transform(X)
X = pd.DataFrame(X, columns=keys)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# Train the best model (assuming CatBoost was the best from previous analysis)
model = CatBoostClassifier(logging_level="Silent", iterations=100, learning_rate=0.1, depth=10)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def get_user_input():
    print("\nPlease enter the following information:")
    user_input = {}
    for column in X.columns:
        while True:
            try:
                value = float(input(f"{column}: "))
                user_input[column] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    return user_input

def predict_liver_disease(user_input):
    input_df = pd.DataFrame([user_input])
    scaled_input = scale.transform(input_df)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0]
    
    result = "Liver disease detected" if prediction[0] == 1 else "No liver disease detected"
    confidence = probability[1] if prediction[0] == 1 else probability[0]
    
    return result, confidence

# Main program loop
while True:
    user_input = get_user_input()
    result, confidence = predict_liver_disease(user_input)
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.2f}")
    
    another = input("\nWould you like to make another prediction? (yes/no): ")
    if another.lower() != 'yes':
        break

print("Thank you for using the Liver Disease Prediction Model.")
