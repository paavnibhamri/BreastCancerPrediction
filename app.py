from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('/Users/aryangupta/college2/projects/breast-cancer-detection/model_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)  # ✅ Correct

@app.route('/')
def home():
    return render_template('index.html')  # this should be your HTML form file

@app.route('/predict', methods=['POST'])
def predict():
 
        # Extract values from form as floats
        input_values = [float(x) for x in request.form.values()]

        # Convert to 2D array for prediction
        input_array = np.array([input_values])
        
        # Predict using the loaded model
        prediction = model.predict(input_array)[0]

        result = 'Has Cancer' if prediction == 1 else 'Not Cancerous'

        return render_template('index.html', prediction=result)
    
  

if __name__ == '__main__':
    app.run(debug=True)
