from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

app = Flask(__name__)

# Check if model files exist; if not, train the model
model_file = 'xgb_model.pkl'
scaler_file = 'scaler.pkl'
encoder_file = 'label_encoders.pkl'

if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(encoder_file)):
    # Load the dataset
    data = pd.read_csv('C:/Users/VAISHNAVI/OneDrive/Desktop/MINI PROJECT/Sheet1.csv')

    # Select relevant features and the target variable
    features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'Gender', 
                'Ethnicity', 'Jundice', 'Autism', 'Country_of_res', 'Used_app_before', 'Relation']
    target = 'Class/ASD'

    # Handle categorical variables
    categorical_features = ['Gender', 'Ethnicity', 'Jundice', 'Autism', 'Country_of_res', 'Used_app_before', 'Relation']
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        data[feature] = label_encoders[feature].fit_transform(data[feature])

    # Drop rows with missing values
    data = data.dropna()

    # Separate the features and the target
    X = data[features]
    y = data[target].apply(lambda x: 1 if x == 'YES' else 0)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the XGBoost model with limited boosting rounds and early stopping
    xgb_model = xgb.XGBClassifier(n_estimators=100, early_stopping_rounds=10, eval_metric="logloss", random_state=42)

    # Train the model with early stopping
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save the model, scaler, and label encoders
    with open(model_file, 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoders, f)
else:
    # Load the model, scaler, and label encoders
    with open(model_file, 'rb') as f:
        xgb_model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_file, 'rb') as f:
        label_encoders = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    new_case = [
        int(form_data['A1_Score']),
        int(form_data['A2_Score']),
        int(form_data['A3_Score']),
        int(form_data['A4_Score']),
        int(form_data['A5_Score']),
        int(form_data['A6_Score']),
        int(form_data['A7_Score']),
        int(form_data['A8_Score']),
        int(form_data['A9_Score']),
        int(form_data['A10_Score']),
        label_encoders['Gender'].transform([form_data['Gender']])[0],
        label_encoders['Ethnicity'].transform([form_data['Ethnicity']])[0],
        label_encoders['Jundice'].transform([form_data['Jundice']])[0],
        label_encoders['Autism'].transform([form_data['Autism']])[0],
        label_encoders['Country_of_res'].transform([form_data['Country_of_res']])[0],
        label_encoders['Used_app_before'].transform([form_data['Used_app_before']])[0],
        label_encoders['Relation'].transform([form_data['Relation']])[0]
    ]

    # Scale the input data
    input_data = scaler.transform([new_case])

    # Make prediction
    prediction = xgb_model.predict(input_data)
    prediction_text = 'Yes' if prediction[0] == 1 else 'No'

    return render_template('index.html', prediction_text=f'Autism Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
