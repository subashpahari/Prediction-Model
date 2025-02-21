from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('rf_model.pkl')
threshold = 0.7032966204148201

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from the form
        features = [
            float(request.form.get('AppendixDiameter')),
            float(request.form.get('ReboundTenderness')),
            float(request.form.get('CoughingPain')),
            float(request.form.get('FreeFluids')),
            float(request.form.get('MigratoryPain')),
            float(request.form.get('BodyTemp')),
            float(request.form.get('KetonesInUrine')),
            float(request.form.get('Nausea')),
            float(request.form.get('WBCCount')),
            float(request.form.get('NeutrophilPerc')),
            float(request.form.get('CRPEntry')),
            float(request.form.get('Peritonitis'))
        ]
        print("features",features)
    except Exception as e:
        return render_template('index.html', prediction_text="Error in input: " + str(e))
    
    # Create a DataFrame with proper column names
    columns = ['AppendixDiameter', 'ReboundTenderness', 'CoughingPain', 'FreeFluids', 'MigratoryPain',
               'BodyTemp', 'KetonesInUrine', 'Nausea', 'WBCCount', 'NeutrophilPerc', 'CRPEntry', 'Peritonitis']
    data = pd.DataFrame([features], columns=columns)
    
    # Get the predicted probability and apply your threshold
    pred_prob = model.predict_proba(data)[0][1]
    prediction = 1 if pred_prob >= threshold else 0
    
    result = "Appendicitis" if prediction == 1 else "No Appendicitis"
    
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
