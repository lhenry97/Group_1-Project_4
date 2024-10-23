from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Step 3: Load the pre-trained logistic regression model and the refitted scaler
model = joblib.load('FeatureReduced_logistic_regression_model.pkl')
scaler = joblib.load('scaler_reduced.pkl')  # Load the new scaler fitted on 11 features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form for the selected 11 features
    input_features = [
        float(request.form['area_mean']),
        float(request.form['concave_points_mean']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['concave_points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['area_se']),
        float(request.form['radius_se'])  
    ]

    # Convert input features into a NumPy array with shape (1, 11)
    features = np.array(input_features).reshape(1, -1)

    # Step 4: Scale the input features using the new refitted scaler
    scaled_features = scaler.transform(features)

    # Make a prediction using the scaled features
    prediction = model.predict(scaled_features)

    # Render the result page with the prediction
    return render_template('result.html', prediction_text=f'The predicted class is: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
