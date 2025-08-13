#!/bin/bash
# setup.sh

# Create directories
mkdir -p models data_cache logs predictions

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database import init_db; init_db()"

# Train initial models
python -c "from training_pipeline import TrainingPipeline; pipeline = TrainingPipeline(); pipeline.run_initial_training()"

# Start services
docker-compose up -d

echo "System ready! Access API at http://localhost:5000"
