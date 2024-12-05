from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logger import log_data
import csv

app = Flask(__name__)


with open('lda_pretrained_model.pkl', 'rb') as f:
    model = pickle.load(f)



with open('columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)



@app.route('/')
def home():
    return render_template('index.html', columns=X_columns)


@app.route('/predict', methods=['POST'])
def predict():
    
    user_input =  {col: request.form[col] for col in X_columns}
    
    
    user_data = pd.DataFrame([user_input])

    
    prediction = model.predict(user_data)

    testvariable= log_data()

    testvariable.log_user_input_and_prediction(user_input, prediction)

    # Example to write user data to CSV
    with open('user_data.csv', mode='a', newline='') as file:
         writer = csv.writer(file)
         if file.tell() == 0:  # Checks if the file is empty
            writer.writerow( list(user_input.keys()) + ["Prediction"])  # Header row
         writer.writerow(list(user_input.values()) +[prediction[0]])
         

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
