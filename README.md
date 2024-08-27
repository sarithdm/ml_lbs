## Overview
This repository contains the code and resources for a Machine Learning session. The session covers setting up the environment, training models, and deploying them using Flask.

## Setup Instructions

1. **Open VS Code and Open a New Terminal:**
   - Launch Visual Studio Code.
   - Open a new terminal within VS Code (\`Terminal > New Terminal\`).

2. **Create a Virtual Environment:**

   python -m venv .mlsession

3. **Activate the Environment:**
   - On Windows:
     .mlsession\\Scripts\\activate
   - On macOS/Linux:
     source .mlsession/bin/activate

4. **Install Essential Packages:**
   pip install numpy pandas scikit-learn tensorflow flask

## Model Deployment with Flask

- **Flask Setup:**
  - Create a Flask application to serve your model.
  - The application will load the trained model, accept input data, and return predictions.

- **Client Code:**
  - A sample Python script for interacting with the Flask API is provided.

## Repository Structure

- app.py: Flask application for serving the model.
- client.py: Python script for testing the Flask API.
- dl_ml.py: Python script for training the model.
- model_IRIS.keras: Trained model file (ensure to include this in the repository).

