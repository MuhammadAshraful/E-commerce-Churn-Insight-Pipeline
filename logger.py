import logging
from datetime import datetime

# Configure logger


class log_data:
              def __init__(self):
                logging.basicConfig(
                  filename='user_activity.log',
                  level=logging.INFO,
                  format='%(asctime)s - %(message)s' 
                  )
              
              
              def log_user_input_and_prediction(self, user_input, prediction):
                 log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_input": user_input,
                "prediction": prediction
                 }
                 logging.info(log_entry)
