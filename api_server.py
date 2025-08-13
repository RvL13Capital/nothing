# api_server.py
"""
REST API für das Breakout Detection System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load configuration
from config_loader import load_config, validate_required_env_vars, get_database_url, get_redis_url

# Validate environment variables
validate_required_env_vars()

app = Flask(__name__)
CORS(app)

# Load application configuration
config = load_config()

# Configure Flask app from configuration
app.config['SECRET_KEY'] = config['security']['secret_key']
app.config['JWT_SECRET_KEY'] = config['security']['jwt_secret_key']
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')
app.config['DATABASE_URL'] = get_database_url()
app.config['REDIS_URL'] = get_redis_url()

# Store config for use in the application
app.config['APP_CONFIG'] = config

# Initialize system
from breakout_system import BreakoutDetectionSystem
system = BreakoutDetectionSystem()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/scan', methods=['POST'])
def scan_tickers():
    """Scan list of tickers for breakouts"""
    data = request.json
    tickers = data.get('tickers', [])
    
    # Run async scan
    loop = asyncio.new_event_loop()
    signals = loop.run_until_complete(system.scan_watchlist(tickers))
    
    return jsonify({
        'signals': [s.to_dict() for s in signals],
        'count': len(signals),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/<ticker>', methods=['GET'])
def analyze_single(ticker):
    """Analyze single ticker"""
    signal = system._analyze_ticker(ticker.upper())
    
    if signal:
        return jsonify(signal.to_dict())
    else:
        return jsonify({'error': 'No signal found'}), 404

@app.route('/train', methods=['POST'])
def trigger_training():
    """Trigger model retraining"""
    # This would trigger async training job
    return jsonify({'status': 'Training started', 'job_id': 'xxx'})

@app.route('/performance', methods=['GET'])
def get_performance():
    """Get system performance metrics"""
    return jsonify(system.performance_metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
