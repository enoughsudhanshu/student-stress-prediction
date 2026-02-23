from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define numerical columns
num_cols = [
    'Age',
    'Daily_Walk_km',
    'Screen_Time_hrs',
    'Sleep_hrs',
    'Study_Work_hrs'
]

cat_cols = [
    'Gender',
    'Activity_Level',
    'Diet_Quality',
    'Caffeine_Intake'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get numerical inputs
        age = float(request.form['Age'])
        walk = float(request.form['Daily_Walk_km'])
        screen = float(request.form['Screen_Time_hrs'])
        sleep = float(request.form['Sleep_hrs'])
        study = float(request.form['Study_Work_hrs'])

        # Get categorical inputs
        gender = request.form['Gender'].lower()
        activity = request.form['Activity_Level'].lower()
        diet = request.form['Diet_Quality'].lower()
        caffeine = request.form['Caffeine_Intake'].lower()

        # Convert categorical manually (IMPORTANT: must match training encoding)
        gender_map = {'male': 1, 'female': 0}
        activity_map = {'low': 1, 'medium': 2, 'high': 0}
        diet_map = {'poor': 2, 'average': 0, 'good': 1}
        caffeine_map = {'low': 1, 'medium': 2, 'high': 0}

        input_data = [
            age,
            walk,
            screen,
            sleep,
            study,
            gender_map.get(gender, 0),
            activity_map.get(activity, 0),
            diet_map.get(diet, 0),
            caffeine_map.get(caffeine, 0)
        ]

        input_array = np.array(input_data).reshape(1, -1)

        # Scale numeric features
        input_array[:, 0:5] = scaler.transform(input_array[:, 0:5])

        prediction = model.predict(input_array)
        result = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"Predicted Stress Level: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)