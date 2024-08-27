import requests
import json

# Define the URL of the Flask application
url = 'http://localhost:5000/predict'

# Define the features you want to predict
data = {
    'features': [5.1, 3.5, 1.4, 0.2]  # Example feature values
}

# Send a POST request to the Flask API
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    print('Prediction:', result['prediction'])
    #print('Probabilities:', result['probabilities'])
else:
    print('Error:', response.status_code, response.text)
