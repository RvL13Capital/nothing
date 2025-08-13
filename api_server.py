# api_server.py
"""
REST API für das Breakout Detection System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from datetime import datetime

app = Flask(__name__)
CORS(app)

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
