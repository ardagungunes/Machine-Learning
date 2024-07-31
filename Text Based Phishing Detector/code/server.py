from flask import Flask, render_template, request
import xgboost as xgb
import pickle
import joblib
import os
from trafilatura import fetch_url, extract
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load your XGBoost model
model = joblib.load("./model/xgb-roberta.pkl")

modelRoberta = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = "Phishing" # or it will be "Legitimate"
    if 'htmlFile' not in request.files:
        return "No file part"
    
    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location
    file_path = 'test/' + file.filename
    file.save(file_path)

    # START of the business logic here
    # Perform prediction using the file_path with your XGBoost model
    # Replace the following line with your actual prediction logic

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            result = extract(file.read())

            embeddings = np.array(modelRoberta.encode(result))
    except:
        with open(file_path, "r", encoding="latin-1") as file:
            result = extract(file.read())

            embeddings = np.array(modelRoberta.encode(result))





    prediction = model.predict(embeddings.reshape(1,-1))


    if prediction == 1:
        prediction_result = "Legitimate"

    # END of the business logic here
   

    return f"{file_path} is {prediction_result}"

if __name__ == '__main__':
    app.run(debug=True, port=5050)