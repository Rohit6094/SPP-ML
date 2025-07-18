Student Performance Prediction System

Overview

This repository contains a machine learning-based system designed to predict student performance (final grade, G3) using a dataset of student-related features. The system employs a Flask web application to provide an interactive dashboard for inputting student details and visualizing predictions from both Random Forest and Linear Regression models, including an ensemble approach.

Features






Data Preprocessing: Handles categorical and numerical data with label encoding and standardization.




Model Training: Utilizes Random Forest Regressor and Linear Regression with GridSearchCV for hyperparameter tuning.



Ensemble Prediction: Combines predictions from both models based on cross-validation MAE weights.



Web Interface: A Flask-based dashboard for user input and result visualization.



Visualization: Includes charts for feature importance, actual vs. predicted grades, and error distribution.



Installation

Prerequisites





Python 3.8+



Required Python packages: numpy, pandas, scikit-learn, flask, joblib



A modern web browser

Setup





Clone the repository:

git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction




Install dependencies:

pip install -r requirements.txt





Ensure the student.csv dataset is in the root directory (sample data included).

Usage





Run the Flask application:

python app.py



Open a web browser and navigate to http://127.0.0.1:5000/.



Fill out the student details form and click "Predict Grade" to see the results.

File Structure





app.py: Main Flask application with model training and prediction logic.



main.js: JavaScript for handling form submission and chart rendering.



index.html: HTML template for the dashboard interface.



static/: Directory for static files (if any).



templates/: Directory containing the index.html template.



feature_columns.joblib, label_encoders.joblib, lr_model.joblib, rf_model.joblib, scaler.joblib: Serialized model and preprocessing objects.



student.csv: Dataset used for training and prediction.

Models





Random Forest Regressor: Tuned with GridSearchCV for optimal performance.



Linear Regression: Provides a baseline model for comparison.



Ensemble: Weighted average based on cross-validation MAE.

Contributing





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Make your changes and commit (git commit -m "Description of changes").



Push to the branch (git push origin feature-branch).



Open a pull request.



Contact

For questions or support, please open an issue on the GitHub repository or contact the maintainer at neurohit223@gmail.com

Acknowledgments





Dataset inspired by student performance studies.



Built with assistance from open-source libraries and community resources.
