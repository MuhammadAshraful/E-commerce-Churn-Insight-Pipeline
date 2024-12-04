from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logger import log_data

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
   

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
