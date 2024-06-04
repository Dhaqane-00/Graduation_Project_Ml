from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load the saved encoders, scaler, and model
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_mode = joblib.load('label_encoder_mode.pkl')
column_transformer = joblib.load('ColumnTransformer.pkl')
sc = joblib.load('StandardScaler.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join('files', file.filename)
    file.save(file_path)

    try:
        # Read the uploaded file as DataFrame
        input_data = pd.read_csv(file_path)

        # Apply Label Encoding to 'Gender' and 'Mode'
        input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
        input_data['Mode'] = label_encoder_mode.transform(input_data['Mode'])

        # Transform the input data using the loaded ColumnTransformer
        transformed_data = column_transformer.transform(input_data)

        # Apply feature scaling
        transformed_data = sc.transform(transformed_data)

        # Make predictions
        predictions = model.predict(transformed_data)

        # Determine the prediction labels
        input_data['Prediction'] = ["Will Graduate" if pred == 1 else "Dropout" for pred in predictions]

        return render_template('index.html', tables=[input_data.to_html(classes='data', header="true")], prediction=True)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)