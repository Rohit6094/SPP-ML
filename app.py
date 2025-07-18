import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
math_df = pd.read_csv('student.csv', sep=';')

# Feature engineering: Add average_grade
math_df['average_grade'] = (math_df['G1'] + math_df['G2']) / 2

# Define preprocessing function
def preprocess(df, label_encoders=None, scaler=None, return_encoders=False):
    numeric = ['age', 'traveltime', 'Medu', 'Fedu', 'studytime', 'failures', 'famrel',
               'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'average_grade']
    cat_nominal = ['Mjob', 'Fjob', 'reason', 'guardian']
    cat_binary = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup',
                  'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    
    if label_encoders is None:
        label_encoders = {}
        for col in cat_binary:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(df[col])
    
    df_encoded = df.copy()
    for col in cat_binary:
        df_encoded[col] = label_encoders[col].transform(df[col])
    
    df_encoded = pd.get_dummies(df_encoded, columns=cat_nominal, prefix=cat_nominal)
    
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_encoded[numeric])
    
    df_encoded[numeric] = scaler.transform(df_encoded[numeric])
    
    if return_encoders:
        return df_encoded, label_encoders, scaler
    return df_encoded, scaler

# Prepare features and target
X = math_df.drop('G3', axis=1)
y = math_df['G3']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess training data and save encoders/scaler
X_train_encoded, label_encoders, scaler = preprocess(X_train, return_encoders=True)
X_test_encoded, _ = preprocess(X_test, label_encoders, scaler)

# Train KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_encoded)
cluster_labels_train = kmeans.predict(X_train_encoded)
cluster_labels_test = kmeans.predict(X_test_encoded)

# Compute cluster characteristics
cluster_characteristics = []
for i in range(n_clusters):
    cluster_data = X_train[cluster_labels_train == i]
    means = cluster_data[['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'average_grade']].mean().to_dict()
    cluster_characteristics.append({k: round(float(v), 2) for k, v in means.items()})

# Train Random Forest with GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5)
rf_model.fit(X_train_encoded, y_train)
print(f"Best RF params: {rf_model.best_params_}")

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train)

# Cross-validation scores
rf_cv_scores = cross_val_score(rf_model.best_estimator_, X_train_encoded, y_train, cv=5, scoring='neg_mean_absolute_error')
lr_cv_scores = cross_val_score(lr_model, X_train_encoded, y_train, cv=5, scoring='neg_mean_absolute_error')
rf_mae_cv = -rf_cv_scores.mean()
lr_mae_cv = -lr_cv_scores.mean()

# Calculate test set metrics
y_pred_rf = rf_model.predict(X_test_encoded)
y_pred_lr = lr_model.predict(X_test_encoded)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Weighted ensemble prediction
weight_rf = lr_mae_cv / (rf_mae_cv + lr_mae_cv)
weight_lr = rf_mae_cv / (rf_mae_cv + lr_mae_cv)

# Save models and encoders
joblib.dump(rf_model.best_estimator_, 'rf_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_train_encoded.columns, 'feature_columns.joblib')

# Load models and encoders
rf_model = joblib.load('rf_model.joblib')
lr_model = joblib.load('lr_model.joblib')
kmeans = joblib.load('kmeans_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert descriptive inputs
        input_data = {}
        for key, value in data.items():
            if key in ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
                mappings = {
                    'traveltime': {'<15 min': 1, '15-30 min': 2, '30 min-1 hour': 3, '>1 hour': 4},
                    'studytime': {'<2 hours': 1, '2-5 hours': 2, '5-10 hours': 3, '>10 hours': 4},
                    'failures': {'0': 0, '1': 1, '2': 2, '3+': 3},
                    'famrel': {'Very Bad': 1, 'Bad': 2, 'Fair': 3, 'Good': 4, 'Very Good': 5},
                    'freetime': {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5},
                    'goout': {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5},
                    'Dalc': {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5},
                    'Walc': {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5},
                    'health': {'Very Bad': 1, 'Bad': 2, 'Fair': 3, 'Good': 4, 'Very Good': 5}
                }
                input_data[key] = mappings[key][value]
            else:
                input_data[key] = float(value) if key in ['age', 'absences', 'G1', 'G2'] else value
        input_data['average_grade'] = (float(input_data.get('G1', 0)) + float(input_data.get('G2', 0))) / 2
        
        input_df = pd.DataFrame([input_data])
        input_encoded, _ = preprocess(input_df, label_encoders, scaler)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        pred_rf = rf_model.predict(input_encoded)[0]
        pred_lr = lr_model.predict(input_encoded)[0]
        pred_ensemble = weight_rf * pred_rf + weight_lr * pred_lr
        cluster_assignment = int(kmeans.predict(input_encoded)[0])
        
        rf_feature_importances = {k: float(v) for k, v in zip(feature_columns, rf_model.feature_importances_)}
        lr_coefficients = {k: float(v) for k, v in zip(feature_columns, lr_model.coef_)}
        
        actual_vs_predicted_rf = [(float(a), float(p)) for a, p in zip(y_test.values, y_pred_rf)]
        actual_vs_predicted_lr = [(float(a), float(p)) for a, p in zip(y_test.values, y_pred_lr)]
        errors_rf = [float(e) for e in (y_test.values - y_pred_rf)]
        errors_lr = [float(e) for e in (y_test.values - y_pred_lr)]
        
        return jsonify({
            'prediction_rf': round(float(pred_rf), 2),
            'prediction_lr': round(float(pred_lr), 2),
            'prediction_ensemble': round(float(pred_ensemble), 2),
            'cluster_assignment': cluster_assignment,
            'cluster_characteristics': cluster_characteristics,
            'mae_rf': round(float(mae_rf), 4),
            'rmse_rf': round(float(rmse_rf), 4),
            'mae_lr': round(float(mae_lr), 4),
            'rmse_lr': round(float(rmse_lr), 4),
            'mae_rf_cv': round(float(rf_mae_cv), 4),
            'mae_lr_cv': round(float(lr_mae_cv), 4),
            'rf_feature_importances': rf_feature_importances,
            'lr_coefficients': lr_coefficients,
            'actual_vs_predicted_rf': actual_vs_predicted_rf,
            'actual_vs_predicted_lr': actual_vs_predicted_lr,
            'errors_rf': errors_rf,
            'errors_lr': errors_lr,
            'cluster_data': {
                'labels': cluster_labels_test.tolist(),
                'features': X_test[['G1', 'studytime']].values.tolist()
            },
            'test_grades': {
                'G3_actual': y_test.tolist(),
                'G3_pred_rf': y_pred_rf.tolist(),
                'G3_pred_lr': y_pred_lr.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)