#!/bin/bash
# setup.sh

echo "🚀 Setting up Breakout Prediction System..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📝 Please copy .env.example to .env and configure your settings:"
    echo "   cp .env.example .env"
    echo ""
    echo "📋 Required environment variables to configure:"
    echo "   - SECRET_KEY (Flask security key)"
    echo "   - POSTGRES_USER, POSTGRES_PASSWORD (Database credentials)"
    echo "   - TWELVEDATA_API_KEY (Market data API key)"
    echo "   - ALPHAVANTAGE_API_KEY (Market data API key)"
    echo "   - REDIS_PASSWORD (Redis security)"
    exit 1
fi

# Load environment variables
source .env

# Create directories
echo "📁 Creating application directories..."
mkdir -p models data_cache logs predictions

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Validate configuration
echo "🔍 Validating configuration..."
python -c "from config_loader import validate_required_env_vars; validate_required_env_vars(); print('✅ Configuration valid')"

# Initialize database
echo "🗄️  Initializing database..."
python -c "from database import init_db; init_db()" || echo "⚠️ Database initialization failed - will retry on first run"

# Train initial models (optional - can be done later)
echo "🧠 Training initial models (this may take a while)..."
python -c "from training_pipeline import TrainingPipeline; pipeline = TrainingPipeline(); pipeline.run_initial_training()" || echo "⚠️ Initial training failed - can be done later via API"

# Start services
echo "🐳 Starting Docker services..."
docker-compose --env-file .env up -d

echo ""
echo "✅ System setup complete!"
echo ""
echo "📊 Access points:"
echo "   - API: http://localhost:${WEB_PORT:-5000}"
echo "   - Health Check: http://localhost:${WEB_PORT:-5000}/health"
echo ""
echo "🔧 To manage services:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
