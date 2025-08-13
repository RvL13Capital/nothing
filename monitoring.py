# monitoring.py
"""
Prometheus Metrics für Monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
predictions_total = Counter('breakout_predictions_total', 'Total predictions made')
prediction_confidence = Histogram('breakout_prediction_confidence', 'Confidence scores')
successful_predictions = Counter('breakout_successful_predictions', 'Successful predictions')
model_performance = Gauge('breakout_model_performance', 'Model performance score')

def setup_monitoring():
    start_http_server(9090)
