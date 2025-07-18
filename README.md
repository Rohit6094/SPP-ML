Student Performance Prediction Dashboard

Overview

Welcome to the Student Performance Prediction Dashboard, an interactive web application designed to predict student final grades (G3) using machine learning models. Built with Flask and modern web technologies, this tool offers a seamless experience for analyzing student data and generating real-time performance predictions.



Student Performance Data Set

Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance Number of Instances: 395 Number of Attributes: 32 Input Features + 1 Target('G3')



Features





Interactive Data Exploration: Visualize student performance distributions, correlations, and key metrics.



Model Comparison: Compare the performance of Random Forest and Linear Regression models.



Real-time Predictions: Input student characteristics and receive instant grade predictions.



Responsive Design: Works seamlessly on both desktop and mobile devices.



Professional Visualizations: Utilize interactive charts for in-depth insights into predictions and errors.



Dashboard Sections

1. Overview





Summary Statistics and Key Metrics: Quick insights into student performance data.



Grade Distribution Histogram: Understand the spread of final grades.



Feature Correlation Analysis: Explore relationships between student features.



Prediction Error Distribution: Analyze the accuracy of model predictions.

2. Analysis





Actual vs. Predicted Grades: Scatter plots showing predicted versus actual grades.



Feature Importance: Visualizations of feature impact on predictions.



Installation

Prerequisites





Python 3.8+



Required packages: flask, numpy, pandas, scikit-learn, joblib

Setup





Clone the repository:

git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction



Install dependencies:

pip install -r requirements.txt

(Note: Create a requirements.txt file with the listed packages if not already present.)



Ensure the student.csv dataset is available in the root directory.

Usage





Run the Flask application:

python app.py



Open your web browser and navigate to http://127.0.0.1:5000.



Input student details via the form and click "Predict Grade" to see results.



File Structure





app.py: Main Flask application with model training and prediction logic.



index.html: Frontend template for the dashboard.



main.js: JavaScript for handling form submission and chart rendering.



templates/: Directory containing the index.html template.



static/: Directory for static files (if any).



feature_columns.joblib, label_encoders.joblib, lr_model.joblib, rf_model.joblib, scaler.joblib: Serialized model and preprocessing objects.



student.csv: Dataset used for training and prediction.



Contributing





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Commit your changes (git commit -m "Add new feature").



Push to the branch (git push origin feature-branch).



Open a pull request.





Contact

For questions or support, please open an issue on GitHub or reach out at neurohit223@gmail.com



Acknowledgments





Inspired by educational performance studies.



Built with contributions from open-source libraries and the developer community.
