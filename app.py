# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model_IRIS.keras')

# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler on the Iris dataset (assuming it is available in the same context)
# If not, fit the scaler with your training data used to train the model
# For demo purposes, we assume you have the Iris dataset available for this

# Dummy Iris data for scaling purposes (replace with actual scaling params)
from sklearn.datasets import load_iris
iris = load_iris()
X_train = iris.data
scaler.fit(X_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert JSON data to a numpy array
        input_data = np.array(data['features']).reshape(1, -1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction_prob = model.predict(input_data_scaled)
        predicted_class = np.argmax(prediction_prob, axis=1)

        # Return the prediction as JSON
        return jsonify({
            'prediction': int(predicted_class[0]),
            'probabilities': prediction_prob[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
