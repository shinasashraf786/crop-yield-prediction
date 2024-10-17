from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model and transformer
#loaded_model = joblib.load('models/knnmodel.joblib')
loaded_model = load_model('models/mlp_model.h5')
#loaded_model = load_model('models/gru_model.h5')
loaded_pt = joblib.load('models/power_transformer.joblib')

# Define dictionaries for dropdown options
crop_dict = {
    'Arecanut': 0, 'Arhar/Tur': 1, 'Bajra': 2, 'Banana': 3, 'Barley': 4, 'Black pepper': 5, 'Cardamom': 6, 'Cashewnut': 7,
    'Castor seed': 8, 'Coconut ': 9, 'Coriander': 10, 'Cotton(lint)': 11, 'Cowpea(Lobia)': 12, 'Dry chillies': 13,
    'Garlic': 14, 'Ginger': 15, 'Gram': 16, 'Groundnut': 17, 'Guar seed': 18, 'Horse-gram': 19, 'Jowar': 20, 'Jute': 21,
    'Khesari': 22, 'Linseed': 23, 'Maize': 24, 'Masoor': 25, 'Mesta': 26, 'Moong(Green Gram)': 27, 'Moth': 28,
    'Niger seed': 29, 'Oilseeds total': 30, 'Onion': 31, 'Other  Rabi pulses': 32, 'Other Cereals': 33,
    'Other Kharif pulses': 34, 'Other Summer Pulses': 35, 'Peas & beans (Pulses)': 36, 'Potato': 37, 'Ragi': 38,
    'Rapeseed &Mustard': 39, 'Rice': 40, 'Safflower': 41, 'Sannhamp': 42, 'Sesamum': 43, 'Small millets': 44,
    'Soyabean': 45, 'Sugarcane': 46, 'Sunflower': 47, 'Sweet potato': 48, 'Tapioca': 49, 'Tobacco': 50, 'Turmeric': 51,
    'Urad': 52, 'Wheat': 53, 'other oilseeds': 54
}

season_dict = {
    'Autumn     ': 0, 'Kharif     ': 1, 'Rabi       ': 2, 'Summer     ': 3, 'Whole Year ': 4, 'Winter     ': 5
}

state_dict = {
    'Andhra Pradesh': 0, 'Arunachal Pradesh': 1, 'Assam': 2, 'Bihar': 3, 'Chhattisgarh': 4, 'Delhi': 5, 'Goa': 6,
    'Gujarat': 7, 'Haryana': 8, 'Himachal Pradesh': 9, 'Jammu and Kashmir': 10, 'Jharkhand': 11, 'Karnataka': 12,
    'Kerala': 13, 'Madhya Pradesh': 14, 'Maharashtra': 15, 'Manipur': 16, 'Meghalaya': 17, 'Mizoram': 18, 'Nagaland': 19,
    'Odisha': 20, 'Puducherry': 21, 'Punjab': 22, 'Sikkim': 23, 'Tamil Nadu': 24, 'Telangana': 25, 'Tripura': 26,
    'Uttar Pradesh': 27, 'Uttarakhand': 28, 'West Bengal': 29
}

@app.route('/')
def home():
    return render_template('index.html', crop_dict=crop_dict, season_dict=season_dict, state_dict=state_dict)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        crop = int(request.form['crop'])
        season = int(request.form['season'])
        state = int(request.form['state'])
        area = float(request.form['area'])
        production = float(request.form['production'])
        annual_rainfall = float(request.form['annual_rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Crop': [crop],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })

        # Transform the input data using the loaded power transformer
        x_test_transformed = loaded_pt.transform(input_data)

        # Make a prediction using the loaded model
        result = loaded_model.predict(x_test_transformed)

        # Get the corresponding crop name from the dictionary
        crop_name = next((name for name, index in crop_dict.items() if index == result[0]), None)

        # Pass the result and crop name to the template
        return render_template('index.html', crop_dict=crop_dict, season_dict=season_dict, state_dict=state_dict, result=result[0][0], crop_name=crop_name)


    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
