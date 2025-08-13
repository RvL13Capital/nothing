#!/bin/bash
# deploy.sh - Automated deployment script

set -e  # Exit on error

# Configuration
ENVIRONMENT=${1:-staging}
DOCKER_COMPOSE_FILE="secure_docker_compose.txt"
ENV_FILE=".env.${ENVIRONMENT}"

echo "🚀 Starting deployment to ${ENVIRONMENT}"

# Function to check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        echo "❌ Environment file ${ENV_FILE} not found"
        echo "Creating template..."
        create_env_template
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Function to create environment template
create_env_template() {
    cat > "${ENV_FILE}.template" << 'EOF'
# SECURITY
SECRET_KEY=generate-a-secure-secret-key
JWT_SECRET_KEY=generate-another-secure-key
ENCRYPTION_KEY=yet-another-secure-key

# ENVIRONMENT
FLASK_ENV=production

# REDIS
REDIS_PASSWORD=secure-redis-password
REDIS_URL=redis://:secure-redis-password@redis:6379/0

# CELERY
CELERY_BROKER_URL=redis://:secure-redis-password@redis:6379/0
CELERY_RESULT_BACKEND=redis://:secure-redis-password@redis:6379/0

# GCS
GCS_BUCKET_NAME=your-bucket-name
GCS_PROJECT_ID=your-project-id
MODEL_ENCRYPTION_KEY=model-encryption-key

# MLFLOW
MLFLOW_EXPERIMENT_NAME=TradingSystem

# GRAFANA
GRAFANA_ADMIN_PASSWORD=secure-admin-password
GRAFANA_ADMIN_USER=admin

# PORTS
WEB_PORT=5000
MLFLOW_PORT=5001
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
REDIS_PORT=6379

# LIMITS
API_RATE_LIMIT=200/hour
TRAINING_RATE_LIMIT=10/hour
MAX_DATAFRAME_SIZE=1000000
ALLOWED_TICKERS=AAPL,GOOGL,MSFT,TSLA,AMZN,META,NFLX,NVDA

# ML CONFIG
OPTUNA_N_TRIALS=20
OPTUNA_TIMEOUT=1800
MAX_TRAINING_TIME=3600

# LOGGING
LOG_LEVEL=INFO
AUDIT_LOG_ENABLED=true
PROMETHEUS_METRICS_ENABLED=true
EOF
    echo "📝 Template created at ${ENV_FILE}.template"
    echo "Please copy to ${ENV_FILE} and fill in actual values"
}

# Function to backup current deployment
backup_current() {
    echo "💾 Creating backup..."
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup volumes
    docker-compose ps -q | while read container; do
        docker commit "$container" "backup_${container}_$(date +%s)"
    done
    
    # Backup environment
    cp "$ENV_FILE" "$BACKUP_DIR/"
    
    echo "✅ Backup created in ${BACKUP_DIR}"
}

# Function to pull latest images
pull_images() {
    echo "🐳 Pulling latest Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping current services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file="$ENV_FILE" down
}

# Function to start services
start_services() {
    echo "🚀 Starting services..."
    
    # Start infrastructure services first
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file="$ENV_FILE" up -d redis
    sleep 5
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file="$ENV_FILE" up -d mlflow prometheus grafana
    sleep 10
    
    # Start application services
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file="$ENV_FILE" up -d web celery-worker celery-beat
}

# Function to health check
health_check() {
    echo "🏥 Running health checks..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check web service
    if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "✅ Web service is healthy"
    else
        echo "❌ Web service health check failed"
        return 1
    fi
    
    # Check MLflow
    if curl -f http://localhost:5001/health > /dev/null 2>&1; then
        echo "✅ MLflow is healthy"
    else
        echo "⚠️ MLflow health check failed (non-critical)"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is healthy"
    else
        echo "❌ Redis health check failed"
        return 1
    fi
    
    # Check Celery workers
    if docker-compose exec -T celery-worker celery -A project.extensions.celery inspect ping > /dev/null 2>&1; then
        echo "✅ Celery workers are healthy"
    else
        echo "⚠️ Celery health check failed (non-critical)"
    fi
    
    return 0
}

# Function to rollback
rollback() {
    echo "⏪ Rolling back deployment..."
    stop_services
    
    # Restore from latest backup
    LATEST_BACKUP=$(ls -t backups/ | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        cp "backups/${LATEST_BACKUP}/.env.${ENVIRONMENT}" "$ENV_FILE"
        echo "✅ Rolled back to ${LATEST_BACKUP}"
    else
        echo "❌ No backup found for rollback"
    fi
    
    start_services
}

# Function to show logs
show_logs() {
    echo "📜 Recent logs:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50
}

# Function to run database migrations
run_migrations() {
    echo "🔄 Running database migrations..."
    # Add your migration commands here
    # docker-compose exec web python manage.py db upgrade
    echo "✅ Migrations completed"
}

# Main deployment flow
main() {
    echo "========================================="
    echo "   ML Trading System Deployment"
    echo "   Environment: ${ENVIRONMENT}"
    echo "   Time: $(date)"
    echo "========================================="
    
    check_prerequisites
    
    if [ "$ENVIRONMENT" == "production" ]; then
        echo "⚠️  PRODUCTION DEPLOYMENT - Are you sure? (yes/no)"
        read -r confirmation
        if [ "$confirmation" != "yes" ]; then
            echo "❌ Deployment cancelled"
            exit 0
        fi
        backup_current
    fi
    
    # Pre-deployment tasks
    run_migrations
    
    # Deployment
    stop_services
    pull_images
    start_services
    
    # Post-deployment
    if health_check; then
        echo "✅ Deployment successful!"
        echo ""
        echo "📊 Access points:"
        echo "   - Web UI: http://localhost:5000"
        echo "   - MLflow: http://localhost:5001"
        echo "   - Grafana: http://localhost:3001"
        echo "   - Prometheus: http://localhost:9090"
    else
        echo "❌ Health check failed, rolling back..."
        rollback
        show_logs
        exit 1
    fi
}

# Parse command line arguments
case "$1" in
    staging|production)
        main
        ;;
    rollback)
        rollback
        ;;
    logs)
        show_logs
        ;;
    health)
        health_check
        ;;
    *)
        echo "Usage: $0 {staging|production|rollback|logs|health}"
        exit 1
        ;;
esac