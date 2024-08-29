from flask import Flask, render_template, request
from model import prediction_model  # Import the PredictionModel instance

app = Flask(__name__)


# Route for the homepage
@app.route('/')
def index():
    numeric_columns_ranges = {
        'CURRENT_ENERGY_RATING': (1, 10),
        'NUMBER_HEATED_ROOMS': (1, 10),
        'NUMBER_HABITABLE_ROOMS': (1, 10),
        'WINDOWS_ENERGY_EFF': (1, 5),
        'ENVIRONMENT_IMPACT_CURRENT': (1, 100),
        'CURRENT_ENERGY_EFFICIENCY': (1, 100),
        'ENERGY_CONSUMPTION_CURRENT': (0, 1000),
        'CO2_EMISSIONS_CURRENT': (0, 1000),
        'LIGHTING_COST_CURRENT': (0, 1000),
        'HEATING_COST_CURRENT': (0, 1000),
        'TOTAL_FLOOR_AREA': (0, 1000)
    }

    return render_template('index.html', numeric_columns_ranges=numeric_columns_ranges)


# Route to handle form submission and predict the result
@app.route('/predict', methods=['POST'])
def predict():
    user_input = {}

    for col in prediction_model.get_columns():
        if col != 'LMK_KEY':  # Skip LMK_KEY
            user_input[col] = [float(request.form.get(col))]  # Convert to float

    prediction = prediction_model.predict(user_input)

    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
    
