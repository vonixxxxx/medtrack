"""
Flask API wrapper for pill prediction
Can be run as a separate service or integrated
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pill_predictor import load_model, predict_from_bytes
import os

app = Flask(__name__)
CORS(app)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pretrained-models', 'cnn2.pth')
model = None

@app.before_first_request
def load_ml_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    
    result = predict_from_bytes(image_bytes, model)
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)



