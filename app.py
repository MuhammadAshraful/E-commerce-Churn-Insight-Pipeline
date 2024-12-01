from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
with open('lda_pretrained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input from the form
    user_input = [float(request.form['feature1']),
                  float(request.form['feature2']),
                  float(request.form['feature3']),
                  # Add more features as needed
                 ]
    
    # Convert input into a DataFrame (ensure it has the correct column names)
    user_data = pd.DataFrame([user_input], columns=['feature1', 'feature2', 'feature3'])

    # Preprocess the input data (apply scaling or encoding if needed)
    # For example, using StandardScaler if your model was trained with scaled features
    scaler = StandardScaler()
    user_data_scaled = scaler.fit_transform(user_data)

    # Make the prediction using the pre-trained model
    prediction = model.predict(user_data_scaled)

    # Return the prediction result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
