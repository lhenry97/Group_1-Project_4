from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('best_logistic_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form for the 29 features
    input_features = [
        float(request.form['radius_mean']),
        float(request.form['texture_mean']),
        float(request.form['perimeter_mean']),
        float(request.form['area_mean']),
        float(request.form['smoothness_mean']),
        float(request.form['compactness_mean']),
        float(request.form['concavity_mean']),
        float(request.form['concave_points_mean']),
        float(request.form['symmetry_mean']),
        float(request.form['fractal_dimension_mean']),
        float(request.form['texture_se']),
        float(request.form['smoothness_se']),
        float(request.form['compactness_se']),
        float(request.form['concave_points_se']),
        float(request.form['symmetry_se']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['compactness_worst']),
        float(request.form['concavity_worst']),
        float(request.form['concave_points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['fractal_dimension_worst']),
        float(request.form['area_se_log']),
        float(request.form['concavity_se_log']),
        float(request.form['perimeter_se_log']),
        float(request.form['radius_se_log']),
        float(request.form['fractal_dimension_se_log'])
    ]

    # Convert input features into a NumPy array with shape (1, 29)
    features = [np.array(input_features)]

    # Make a prediction
    prediction = model.predict(features)

    # Render the result page
    return render_template('result.html', prediction_text=f'The predicted class is: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)

