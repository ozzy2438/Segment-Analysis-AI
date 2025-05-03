# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
import tempfile
import time
from werkzeug.utils import secure_filename
import numpy as np

from data_analyzer import DataAnalyzer
from segmentation_model import DynamicSegmentationModel
from segment_analyzer import SegmentAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize with your OpenAI API key
OPENAI_API_KEY = "your-openai-api-key"  # Normally you'd store this securely, e.g. in environment variables

# Initialize components
data_analyzer = DataAnalyzer(api_key=OPENAI_API_KEY)
segment_analyzer = SegmentAnalyzer(api_key=OPENAI_API_KEY)
segmentation_model = DynamicSegmentationModel()

# Global variables to store session data
current_data = None
analysis_results = None
segment_profiles = None
segment_analysis = None
campaign_strategy = None
data_hash = None  # To detect changes in data

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global current_data, analysis_results, segment_profiles, segment_analysis, campaign_strategy, data_hash
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file to temporary location
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(file_path)
    
    try:
        # Determine file type and read data
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            current_data = pd.read_csv(file_path)
        elif file_extension in ['xlsx', 'xls']:
            current_data = pd.read_excel(file_path)
        elif file_extension == 'json':
            current_data = pd.read_json(file_path)
        else:
            return jsonify({'error': f'Unsupported file type: {file_extension}'}), 400
        
        # Generate a hash of the data to detect changes
        new_data_hash = hash(str(current_data.head(10)))
        
        # If this is new data, reset all analysis results
        if new_data_hash != data_hash:
            data_hash = new_data_hash
            analysis_results = None
            segment_profiles = None
            segment_analysis = None
            campaign_strategy = None
            
            # Analyze the new data with OpenAI
            analysis_results = data_analyzer.analyze_dataset(current_data)
            
            # Fit segmentation model based on AI recommendations
            segmentation_model.fit(current_data, analysis_results["ai_analysis"])
            
            # Update current data with segment assignments
            current_data = segmentation_model.predict(current_data)
            
            # Get segment profiles
            segment_profiles = segmentation_model.segment_profiles
            
            # Analyze segments with OpenAI
            segment_analysis = segment_analyzer.analyze_segments(
                segment_profiles, 
                current_data, 
                analysis_results["ai_analysis"]
            )
            
            # Generate overall campaign strategy
            campaign_strategy = segment_analyzer.generate_campaign_ideas(segment_analysis)
            
        return jsonify({
            'success': True,
            'message': 'New file processed successfully',
            'rows': len(current_data),
            'columns': len(current_data.columns),
            'segments': len(segment_profiles) if segment_profiles else 0,
            'analysis_complete': analysis_results is not None,
            'data_preview': current_data.head(10).to_dict(orient='records')
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e), 
            'traceback': traceback.format_exc()
        }), 500
    finally:
        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/data/preview', methods=['GET'])
def get_data_preview():
    if current_data is None:
        return jsonify({'error': 'No data loaded yet'}), 400
    
    return jsonify(current_data.head(10).to_dict(orient='records'))