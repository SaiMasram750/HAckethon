from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import io
import os
import uuid
import datetime
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Try to import MNE, but provide fallback if not available
try:
    import mne
    MNE_AVAILABLE = True
    print("✅ MNE library available for EDF processing")
except ImportError:
    MNE_AVAILABLE = False
    print("⚠ MNE library not found. EDF processing will be limited.")
    print("Install MNE with: pip install mne")

app = Flask(__name__)

# Enable CORS for all routes (allows frontend from different domains)
CORS(app)

# API Configuration
API_CONFIG = {
    'version': '1.0.0',
    'name': 'Schizophrenia Detection API',
    'description': 'EEG-based schizophrenia detection system',
    'base_url': 'http://localhost:5000/api/v1'
}

# Simple API key system (in production, use proper authentication)
API_KEYS = {
    'dev-key-123': {'name': 'Development Key', 'permissions': ['read', 'write']},
    'frontend-key-456': {'name': 'Frontend Key', 'permissions': ['read', 'write']},
    'test-key-789': {'name': 'Test Key', 'permissions': ['read']}
}

def validate_api_key(api_key):
    """Validate API key"""
    return api_key in API_KEYS

def require_api_key(f):
    """Decorator to require API key for endpoints"""
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Global variables
model = None
scaler = StandardScaler()
MODEL_PATH = 'schizophrenia_detection_model.h5'  # Update this path

# Configuration for EEG processing
EEG_CONFIG = {
    'sampling_rate': 256,  # Adjust based on your data
    'window_size': 1000,   # Number of samples per window
    'overlap': 0.5,        # 50% overlap
    'frequency_bands': {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
}

# HTML template for schizophrenia detection
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Schizophrenia Detection - EEG Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #3498db; border-radius: 10px; }
        .file-input { margin: 10px 0; padding: 10px; }
        label { font-weight: bold; color: #34495e; display: block; margin: 10px 0 5px 0; }
        input[type="file"] { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100%; }
        select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100%; }
        button { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
        button:hover { background: #2980b9; }
        .result { margin-top: 20px; padding: 20px; border-radius: 8px; }
        .error { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .success { background: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9; }
        .warning { background: #fff3e0; color: #f57c00; border: 1px solid #ffcc02; }
        .info-section { background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .prediction-result { font-size: 18px; font-weight: bold; text-align: center; padding: 20px; }
        .probability-bar { background: #ecf0f1; height: 20px; border-radius: 10px; margin: 10px 0; position: relative; }
        .probability-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; }
        .normal { background: #27ae60; }
        .risk { background: #e74c3c; }
        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Schizophrenia Detection System</h1>
        
        <div class="info-section">
            <h3>📋 Instructions:</h3>
            <ul>
                <li><strong>EDF Files:</strong> Upload EEG recordings in European Data Format</li>
                <li><strong>CSV Files:</strong> Upload preprocessed EEG features or raw signal data</li>
                <li>Supported formats: .edf, .csv</li>
                <li>The system analyzes EEG patterns associated with schizophrenia indicators</li>
            </ul>
        </div>
        
        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input">
                    <label for="edfFile">EEG Data File (EDF/CSV):</label>
                    <input type="file" id="fileInput" name="file" accept=".edf,.csv" required>
                </div>
                
                <div class="file-input">
                    <label for="analysisType">Analysis Type:</label>
                    <select id="analysisType" name="analysis_type">
                        <option value="full">Full Analysis (Recommended)</option>
                        <option value="quick">Quick Screening</option>
                        <option value="detailed">Detailed Report</option>
                    </select>
                </div>
                
                <button type="submit">🔬 Analyze EEG Data</button>
            </form>
        </div>
        
        <div id="result"></div>
        
        <div class="info-section">
            <h3>🔬 API Endpoints:</h3>
            <ul>
                <li><strong>POST /analyze/edf</strong> - Analyze EDF file</li>
                <li><strong>POST /analyze/csv</strong> - Analyze CSV file</li>
                <li><strong>POST /predict</strong> - General prediction endpoint</li>
                <li><strong>GET /model/info</strong> - Model information</li>
                <li><strong>GET /health</strong> - System health check</li>
            </ul>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const analysisType = document.getElementById('analysisType');
            const resultDiv = document.getElementById('result');
            
            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<div class="result error">❌ Please select a file first.</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('analysis_type', analysisType.value);
            
            const fileName = fileInput.files[0].name.toLowerCase();
            let endpoint = '/predict';
            
            if (fileName.endsWith('.edf')) {
                endpoint = '/analyze/edf';
            } else if (fileName.endsWith('.csv')) {
                endpoint = '/analyze/csv';
            }
            
            resultDiv.innerHTML = '<div class="result">🔄 Processing EEG data... This may take a few moments.</div>';
            
            fetch(endpoint, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = '<div class="result error">❌ Error: ' + data.error + '</div>';
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                resultDiv.innerHTML = '<div class="result error">❌ Analysis failed: ' + error.message + '</div>';
            });
        };
        
        function displayResults(data) {
            const resultDiv = document.getElementById('result');
            const probability = data.schizophrenia_probability || data.max_probability || 0;
            const isRisk = probability > 0.5;
            
            let html = '<div class="result success">';
            html += '<div class="prediction-result">';
            html += isRisk ? '⚠ Elevated Risk Detected' : '✅ Normal Pattern';
            html += '</div>';
            
            html += '<div class="probability-bar">';
            html += '<div class="probability-fill ' + (isRisk ? 'risk' : 'normal') + '" style="width: ' + (probability * 100) + '%"></div>';
            html += '</div>';
            html += '<p style="text-align: center;">Probability: ' + (probability * 100).toFixed(1) + '%</p>';
            
            if (data.features) {
                html += '<h4>📊 Extracted Features:</h4>';
                html += '<ul>';
                for (const [key, value] of Object.entries(data.features)) {
                    html += '<li><strong>' + key + ':</strong> ' + (typeof value === 'number' ? value.toFixed(4) : value) + '</li>';
                }
                html += '</ul>';
            }
            
            if (data.analysis_details) {
                html += '<h4>🔍 Analysis Details:</h4>';
                html += '<pre>' + JSON.stringify(data.analysis_details, null, 2) + '</pre>';
            }
            
            html += '</div>';
            resultDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

def load_model():
    """Load the schizophrenia detection model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Schizophrenia detection model loaded from {MODEL_PATH}")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            model = None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None

def extract_eeg_features(eeg_data, sampling_rate=256):
    """Extract features from EEG data for schizophrenia detection"""
    features = {}
    
    try:
        # Ensure eeg_data is 2D (channels, timepoints)
        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        # Statistical features
        features['mean'] = np.mean(eeg_data, axis=1)
        features['std'] = np.std(eeg_data, axis=1)
        features['variance'] = np.var(eeg_data, axis=1)
        features['skewness'] = pd.DataFrame(eeg_data.T).skew().values
        features['kurtosis'] = pd.DataFrame(eeg_data.T).kurtosis().values
        
        # Frequency domain features
        for channel_idx in range(min(eeg_data.shape[0], 8)):  # Limit to first 8 channels
            channel_data = eeg_data[channel_idx]
            
            if len(channel_data) > 256:  # Need enough data for frequency analysis
                # Power spectral density
                freqs, psd = signal.welch(channel_data, sampling_rate, nperseg=min(len(channel_data)//4, 256))
                
                # Band power features
                for band_name, (low_freq, high_freq) in EEG_CONFIG['frequency_bands'].items():
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        features[f'{band_name}_power_ch{channel_idx}'] = band_power
        
        # Connectivity features (simplified)
        if eeg_data.shape[0] > 1:
            correlation_matrix = np.corrcoef(eeg_data)
            features['avg_connectivity'] = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            features['max_connectivity'] = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        
        # Complexity features
        for channel_idx in range(min(eeg_data.shape[0], 5)):  # Limit to first 5 channels
            channel_data = eeg_data[channel_idx]
            features[f'complexity_ch{channel_idx}'] = np.std(np.diff(channel_data))
            
    except Exception as e:
        print(f"Warning: Error extracting some features: {str(e)}")
    
    return features

def process_edf_file(file_content):
    """Process EDF file and extract EEG data"""
    if not MNE_AVAILABLE:
        raise Exception("MNE library is required for EDF file processing. Install with: pip install mne")
    
    try:
        # Save temporary file
        temp_path = 'temp_eeg.edf'
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        # Load EDF file using MNE
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
        
        # Get EEG data
        eeg_data = raw.get_data()  # Shape: (n_channels, n_timepoints)
        sampling_rate = raw.info['sfreq']
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Basic preprocessing - Apply bandpass filter (0.5-100 Hz)
        if eeg_data.shape[1] > 100:  # Only filter if we have enough samples
            eeg_data = butter_bandpass_filter(eeg_data, 0.5, min(100, sampling_rate/2-1), sampling_rate)
        
        return eeg_data, sampling_rate, raw.info
        
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists('temp_eeg.edf'):
            os.remove('temp_eeg.edf')
        raise Exception(f"Error processing EDF file: {str(e)}")

def process_csv_file(file_content):
    """Process CSV file containing EEG data or features"""
    try:
        # Read CSV file
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Check if this is raw EEG data or extracted features
        if len(df.columns) > 20 or any('channel' in col.lower() or 'eeg' in col.lower() for col in df.columns):
            # Assume this is raw EEG data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                eeg_data = df[numeric_columns].values.T  # Transpose to (channels, timepoints)
                return eeg_data, EEG_CONFIG['sampling_rate'], 'raw_csv'
            else:
                raise Exception("No numeric data found in CSV file")
        else:
            # Assume this is feature data
            numeric_data = df.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                raise Exception("No numeric features found in CSV file")
            return numeric_data.values, None, 'features_csv'
            
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to EEG data"""
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure frequencies are valid
        if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
            print("Warning: Invalid filter frequencies, skipping filtering")
            return data
            
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=1)
    except Exception as e:
        print(f"Warning: Filtering failed: {str(e)}, returning original data")
        return data

@app.route('/api/v1/docs')
def api_docs():
    """API Documentation endpoint"""
    docs = {
        'api_info': API_CONFIG,
        'authentication': {
            'method': 'API Key',
            'header': 'X-API-Key',
            'parameter': 'api_key',
            'example_keys': {
                'development': 'dev-key-123',
                'frontend': 'frontend-key-456',
                'testing': 'test-key-789'
            }
        },
        'endpoints': {
            'GET /api/v1/health': {
                'description': 'System health check',
                'auth_required': False,
                'response': {
                    'status': 'healthy',
                    'model_loaded': 'boolean',
                    'timestamp': 'ISO datetime'
                }
            },
            'GET /api/v1/model/info': {
                'description': 'Get model information',
                'auth_required': True,
                'response': {
                    'model_type': 'schizophrenia_detection',
                    'input_shape': 'string',
                    'output_shape': 'string'
                }
            },
            'POST /api/v1/analyze': {
                'description': 'Analyze EEG file for schizophrenia detection',
                'auth_required': True,
                'content_type': 'multipart/form-data',
                'parameters': {
                    'file': 'EEG file (.edf or .csv)',
                    'analysis_type': 'optional: quick/full/detailed (default: full)'
                },
                'response': {
                    'prediction_id': 'unique identifier',
                    'schizophrenia_probability': 'float 0-1',
                    'risk_level': 'LOW/MODERATE/HIGH',
                    'features': 'extracted features object',
                    'timestamp': 'ISO datetime'
                }
            },
            'POST /api/v1/batch-analyze': {
                'description': 'Batch analyze multiple files',
                'auth_required': True,
                'content_type': 'multipart/form-data',
                'parameters': {
                    'files': 'Multiple EEG files'
                }
            }
        },
        'examples': {
            'curl_health': 'curl -X GET http://localhost:5000/api/v1/health',
            'curl_analyze': 'curl -X POST -H "X-API-Key: frontend-key-456" -F "file=@eeg_data.edf" http://localhost:5000/api/v1/analyze',
            'javascript_fetch': '''
fetch('http://localhost:5000/api/v1/analyze', {
    method: 'POST',
    headers: {
        'X-API-Key': 'frontend-key-456'
    },
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
            '''
        }
    }
    return jsonify(docs)

@app.route('/api/v1/health')
def api_health():
    """API Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'schizophrenia_detection_api',
        'version': API_CONFIG['version'],
        'supported_formats': ['edf', 'csv'],
        'mne_available': MNE_AVAILABLE,
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/v1/model/info')
@require_api_key
def api_model_info():
    """Get model information via API"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    return jsonify({
        'model_type': 'schizophrenia_detection',
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'supported_formats': ['EDF', 'CSV'],
        'eeg_config': EEG_CONFIG,
        'mne_available': MNE_AVAILABLE,
        'version': API_CONFIG['version'],
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/v1/analyze', methods=['POST'])
@require_api_key
def api_analyze():
    """Main API endpoint for EEG analysis"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'code': 'MODEL_NOT_LOADED'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'code': 'NO_FILE'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'code': 'EMPTY_FILE'}), 400
    
    analysis_type = request.form.get('analysis_type', 'full')
    prediction_id = str(uuid.uuid4())
    
    try:
        filename = file.filename.lower()
        
        # Route to appropriate analyzer
        if filename.endswith('.edf'):
            result = process_edf_analysis(file, analysis_type, prediction_id)
        elif filename.endswith('.csv'):
            result = process_csv_analysis(file, analysis_type, prediction_id)
        else:
            return jsonify({
                'error': 'Unsupported file format. Please upload EDF or CSV files.',
                'code': 'UNSUPPORTED_FORMAT',
                'supported_formats': ['edf', 'csv']
            }), 400
        
        # Add API metadata
        result.update({
            'prediction_id': prediction_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'api_version': API_CONFIG['version'],
            'processing_time': 'completed'
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'code': 'ANALYSIS_FAILED',
            'prediction_id': prediction_id,
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/v1/batch-analyze', methods=['POST'])
@require_api_key
def api_batch_analyze():
    """Batch analysis endpoint for multiple files"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'code': 'MODEL_NOT_LOADED'}), 500
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded', 'code': 'NO_FILES'}), 400
    
    results = []
    batch_id = str(uuid.uuid4())
    
    for i, file in enumerate(files):
        if file.filename == '':
            continue
            
        try:
            prediction_id = f"{batch_id}-{i}"
            filename = file.filename.lower()
            
            if filename.endswith('.edf'):
                result = process_edf_analysis(file, 'quick', prediction_id)
            elif filename.endswith('.csv'):
                result = process_csv_analysis(file, 'quick', prediction_id)
            else:
                result = {
                    'error': 'Unsupported format',
                    'filename': file.filename
                }
            
            result.update({
                'filename': file.filename,
                'prediction_id': prediction_id
            })
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'prediction_id': f"{batch_id}-{i}"
            })
    
    return jsonify({
        'batch_id': batch_id,
        'total_files': len(files),
        'processed_files': len(results),
        'results': results,
        'timestamp': datetime.datetime.now().isoformat()
    })

def process_edf_analysis(file, analysis_type, prediction_id):
    """Process EDF file analysis for API"""
    file_content = file.read()
    eeg_data, sampling_rate, eeg_info = process_edf_file(file_content)
    
    features = extract_eeg_features(eeg_data, sampling_rate)
    
    # Prepare features for model prediction
    feature_vector = []
    for key in sorted(features.keys()):
        if isinstance(features[key], np.ndarray):
            feature_vector.extend(features[key].flatten())
        else:
            feature_vector.append(features[key])
    
    feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    feature_vector = np.nan_to_num(feature_vector)
    feature_vector = scaler.fit_transform(feature_vector)
    
    # Adjust feature vector size
    if feature_vector.shape[1] != model.input_shape[1]:
        if feature_vector.shape[1] < model.input_shape[1]:
            padding = np.zeros((1, model.input_shape[1] - feature_vector.shape[1]))
            feature_vector = np.concatenate([feature_vector, padding], axis=1)
        else:
            feature_vector = feature_vector[:, :model.input_shape[1]]
    
    prediction = model.predict(feature_vector, verbose=0)
    probability = float(prediction[0][0]) if len(prediction[0]) == 1 else float(np.max(prediction))
    
    result = {
        'schizophrenia_probability': probability,
        'risk_level': 'HIGH' if probability > 0.7 else 'MODERATE' if probability > 0.3 else 'LOW',
        'confidence_score': float(1 - abs(0.5 - probability) * 2),  # How confident the model is
        'file_info': {
            'type': 'edf',
            'channels': int(eeg_data.shape[0]),
            'duration_seconds': float(eeg_data.shape[1] / sampling_rate),
            'sampling_rate': float(sampling_rate)
        },
        'analysis_type': analysis_type
    }
    
    if analysis_type in ['full', 'detailed']:
        result['features'] = {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() if isinstance(v, np.ndarray) else str(v) 
                            for k, v in list(features.items())[:15]}
    
    if analysis_type == 'detailed':
        result['detailed_features'] = {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() if isinstance(v, np.ndarray) else str(v) 
                                     for k, v in features.items()}
    
    return result

def process_csv_analysis(file, analysis_type, prediction_id):
    """Process CSV file analysis for API"""
    file_content = file.read()
    data, sampling_rate, data_type = process_csv_file(file_content)
    
    if data_type == 'raw_csv':
        features = extract_eeg_features(data, sampling_rate)
        
        feature_vector = []
        for key in sorted(features.keys()):
            if isinstance(features[key], np.ndarray):
                feature_vector.extend(features[key].flatten())
            else:
                feature_vector.append(features[key])
        
        feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    else:
        feature_vector = data.reshape(1, -1) if len(data.shape) == 1 else data
        features = {f'feature_{i}': feature_vector[0][i] for i in range(min(10, feature_vector.shape[1]))}
    
    feature_vector = np.nan_to_num(feature_vector)
    feature_vector = scaler.fit_transform(feature_vector)
    
    # Adjust feature vector size
    if feature_vector.shape[1] != model.input_shape[1]:
        if feature_vector.shape[1] < model.input_shape[1]:
            padding = np.zeros((1, model.input_shape[1] - feature_vector.shape[1]))
            feature_vector = np.concatenate([feature_vector, padding], axis=1)
        else:
            feature_vector = feature_vector[:, :model.input_shape[1]]
    
    prediction = model.predict(feature_vector, verbose=0)
    probability = float(prediction[0][0]) if len(prediction[0]) == 1 else float(np.max(prediction))
    
    result = {
        'schizophrenia_probability': probability,
        'risk_level': 'HIGH' if probability > 0.7 else 'MODERATE' if probability > 0.3 else 'LOW',
        'confidence_score': float(1 - abs(0.5 - probability) * 2),
        'file_info': {
            'type': 'csv',
            'data_type': data_type
        },
        'analysis_type': analysis_type
    }
    
    if analysis_type in ['full', 'detailed']:
        result['features'] = features
    
    return result

@app.route('/')
def home():
    """Serve the main interface"""
    return render_template_string(HTML_TEMPLATE)

# Keep existing endpoints for backward compatibility
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'schizophrenia_detection',
        'supported_formats': ['edf', 'csv'],
        'mne_available': MNE_AVAILABLE
    })

@app.route('/model/info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    return jsonify({
        'model_type': 'schizophrenia_detection',
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'supported_formats': ['EDF', 'CSV'],
        'eeg_config': EEG_CONFIG,
        'mne_available': MNE_AVAILABLE
    })

@app.route('/analyze/edf', methods=['POST'])
def analyze_edf():
    """Analyze EDF file for schizophrenia detection"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No EDF file uploaded'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('analysis_type', 'full')
    
    try:
        # Process EDF file
        file_content = file.read()
        eeg_data, sampling_rate, eeg_info = process_edf_file(file_content)
        
        # Extract features
        features = extract_eeg_features(eeg_data, sampling_rate)
        
        # Prepare features for model prediction
        feature_vector = []
        for key in sorted(features.keys()):
            if isinstance(features[key], np.ndarray):
                feature_vector.extend(features[key].flatten())
            else:
                feature_vector.append(features[key])
        
        feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector)
        
        # Normalize features
        feature_vector = scaler.fit_transform(feature_vector)
        
        # Make prediction
        if feature_vector.shape[1] != model.input_shape[1]:
            # Pad or truncate to match model input
            if feature_vector.shape[1] < model.input_shape[1]:
                padding = np.zeros((1, model.input_shape[1] - feature_vector.shape[1]))
                feature_vector = np.concatenate([feature_vector, padding], axis=1)
            else:
                feature_vector = feature_vector[:, :model.input_shape[1]]
        
        prediction = model.predict(feature_vector, verbose=0)
        probability = float(prediction[0][0]) if len(prediction[0]) == 1 else float(np.max(prediction))
        
        result = {
            'schizophrenia_probability': probability,
            'risk_level': 'HIGH' if probability > 0.7 else 'MODERATE' if probability > 0.3 else 'LOW',
            'features': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() if isinstance(v, np.ndarray) else str(v) 
                        for k, v in list(features.items())[:10]},  # Limit features for display
            'eeg_info': {
                'channels': eeg_data.shape[0],
                'duration': eeg_data.shape[1] / sampling_rate,
                'sampling_rate': sampling_rate
            },
            'analysis_type': analysis_type
        }
        
        if analysis_type == 'detailed':
            result['detailed_features'] = {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() if isinstance(v, np.ndarray) else str(v) 
                                         for k, v in features.items()}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'EDF analysis failed: {str(e)}'}), 500

@app.route('/analyze/csv', methods=['POST'])
def analyze_csv():
    """Analyze CSV file for schizophrenia detection"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No CSV file uploaded'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('analysis_type', 'full')
    
    try:
        # Process CSV file
        file_content = file.read()
        data, sampling_rate, data_type = process_csv_file(file_content)
        
        if data_type == 'raw_csv':
            # Extract features from raw EEG data
            features = extract_eeg_features(data, sampling_rate)
            
            # Prepare features for model
            feature_vector = []
            for key in sorted(features.keys()):
                if isinstance(features[key], np.ndarray):
                    feature_vector.extend(features[key].flatten())
                else:
                    feature_vector.append(features[key])
            
            feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
            
        else:
            # Use pre-extracted features
            feature_vector = data.reshape(1, -1) if len(data.shape) == 1 else data
            features = {f'feature_{i}': feature_vector[0][i] for i in range(min(10, feature_vector.shape[1]))}
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector)
        
        # Normalize features
        feature_vector = scaler.fit_transform(feature_vector)
        
        # Adjust feature vector size to match model input
        if feature_vector.shape[1] != model.input_shape[1]:
            if feature_vector.shape[1] < model.input_shape[1]:
                padding = np.zeros((1, model.input_shape[1] - feature_vector.shape[1]))
                feature_vector = np.concatenate([feature_vector, padding], axis=1)
            else:
                feature_vector = feature_vector[:, :model.input_shape[1]]
        
        prediction = model.predict(feature_vector, verbose=0)
        probability = float(prediction[0][0]) if len(prediction[0]) == 1 else float(np.max(prediction))
        
        result = {
            'schizophrenia_probability': probability,
            'risk_level': 'HIGH' if probability > 0.7 else 'MODERATE' if probability > 0.3 else 'LOW',
            'features': features,
            'data_type': data_type,
            'analysis_type': analysis_type
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'CSV analysis failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """General prediction endpoint that routes to appropriate analyzer"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = file.filename.lower()
    
    # Route to appropriate analyzer based on file extension
    if filename.endswith('.edf'):
        return analyze_edf()
    elif filename.endswith('.csv'):
        return analyze_csv()
    else:
        return jsonify({'error': 'Unsupported file format. Please upload EDF or CSV files for schizophrenia detection.'}), 400

if __name__ == '__main__':
    print("🧠 Starting Schizophrenia Detection API Server...")
    print("📊 Supported formats: EDF, CSV")
    print("🔗 API Base URL: http://localhost:5000/api/v1")
    print("📚 API Documentation: http://localhost:5000/api/v1/docs")
    print("🔑 API Keys:")
    for key, info in API_KEYS.items():
        print(f"   - {key}: {info['name']}")
    print("🔬 Loading model...")
    
    load_model()
    
    if model is None:
        print("⚠  Warning: Model not loaded. Please check the model path.")
    else:
        print("✅ Model loaded successfully!")
    
    print("🌐 CORS enabled for cross-origin requests")
    print("🚀 Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
