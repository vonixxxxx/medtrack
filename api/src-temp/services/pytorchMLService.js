const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * PyTorch ML Model Service
 * Based on Confir-Med livemodel.py
 * Integrates with Python ML model for pill recognition
 */

// Path to Python ML service
const PYTHON_SCRIPT_PATH = path.join(__dirname, '../../ml-service/predict.py');
const MODEL_PATH = path.join(__dirname, '../../ml-service/pretrained-models/cnn2.pth');

// Medicine classes from Confir-Med
const MEDICINES = [
  'Alaxan',
  'Bactidol',
  'Bioflu',
  'Biogesic',
  'DayZinc',
  'Fish Oil',
  'Kremil S',
  'Medicol',
  'Neozep'
];

/**
 * Check if ML service is available
 */
function isMLServiceAvailable() {
  return fs.existsSync(PYTHON_SCRIPT_PATH) && fs.existsSync(MODEL_PATH);
}

/**
 * Predict medication from image using Python ML model
 * This calls the Python script that uses the PyTorch model
 */
async function predictWithMLModel(imagePath) {
  return new Promise((resolve, reject) => {
    if (!isMLServiceAvailable()) {
      // Fallback to basic recognition if ML service not available
      return resolve({
        predicted_medicine: 'Unknown',
        confidence: 0.5,
        available: false
      });
    }

    const pythonProcess = spawn('python3', [
      PYTHON_SCRIPT_PATH,
      imagePath,
      MODEL_PATH
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python ML process error:', stderr);
        reject(new Error(`ML prediction failed: ${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout);
        resolve({
          predicted_medicine: result.predicted_medicine || 'Unknown',
          confidence: result.confidence || 0.5,
          available: true
        });
      } catch (error) {
        console.error('Error parsing ML result:', error);
        reject(new Error('Failed to parse ML prediction result'));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('Error spawning Python process:', error);
      reject(error);
    });
  });
}

/**
 * Alternative: Call external ML API (if deployed separately)
 */
async function predictWithAPI(imagePath) {
  try {
    const FormData = require('form-data');
    const fs = require('fs');
    const fetch = require('node-fetch');

    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`ML API error: ${response.statusText}`);
    }

    const result = await response.json();
    return {
      predicted_medicine: result.predicted_medicine,
      confidence: parseFloat(result.confidence.replace('%', '')) / 100,
      available: true
    };
  } catch (error) {
    console.error('Error calling ML API:', error);
    throw error;
  }
}

module.exports = {
  predictWithMLModel,
  predictWithAPI,
  isMLServiceAvailable,
  MEDICINES
};



