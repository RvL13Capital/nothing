# cloud_deployment.py
"""
Google Cloud deployment configuration for the 40%+ Breakout Prediction System
Ultra-low cost serverless deployment
"""

import os
import json
from typing import Dict, List, Any

def create_cloud_function_config() -> Dict[str, Any]:
    """Create Google Cloud Function configuration"""
    
    return {
        "name": "breakout-predictor",
        "description": "40%+ Breakout Prediction AI for micro/small cap stocks",
        "runtime": "python311",
        "available_memory_mb": 512,  # Minimal memory for cost efficiency
        "timeout": "540s",  # 9 minutes max
        "entry_point": "cloud_function_entry_point",
        "environment_variables": {
            "GCS_BUCKET_NAME": "${GCS_BUCKET_NAME}",
            "TWELVEDATA_API_KEY": "${TWELVEDATA_API_KEY}",
            "ALPHAVANTAGE_API_KEY": "${ALPHAVANTAGE_API_KEY}"
        },
        "trigger": {
            "http_trigger": {
                "security_level": "SECURE_ALWAYS"
            }
        },
        "ingress_settings": "ALLOW_ALL"
    }

def create_cloud_scheduler_config() -> Dict[str, Any]:
    """Create Cloud Scheduler job for daily EOD processing"""
    
    return {
        "name": "daily-breakout-scan",
        "description": "Daily scan for 40%+ breakout opportunities",
        "schedule": "0 18 * * 1-5",  # 6 PM EST, weekdays only
        "timezone": "America/New_York",
        "http_target": {
            "uri": "${CLOUD_FUNCTION_URL}",
            "http_method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "tickers": [
                    # $300M-$2B market cap watchlist (higher quality stocks)
                    "PLTR", "SOFI", "HOOD", "AFRM", "UPST", "ROKU", "SNAP",
                    "PTON", "BYND", "CRSR", "DKNG", "FUBO", "SPCE", "OPEN",
                    "SKLZ", "COIN", "RBLX", "ABNB", "DASH", "SNOW", "NET"
                ]
            }).encode()
        }
    }

def create_gcs_bucket_config() -> Dict[str, Any]:
    """Create GCS bucket configuration"""
    
    return {
        "name": "${GCS_BUCKET_NAME}",
        "location": "US-CENTRAL1",  # Cheapest region
        "storage_class": "STANDARD",
        "public_access_prevention": "inherited",
        "uniform_bucket_level_access": False,  # Allow individual file permissions
        "lifecycle": {
            "rule": [
                {
                    "action": {"type": "Delete"},
                    "condition": {
                        "age": 90,  # Delete files older than 90 days
                        "matches_prefix": ["data/raw/", "logs/"]
                    }
                },
                {
                    "action": {"type": "SetStorageClass", "storage_class": "NEARLINE"},
                    "condition": {
                        "age": 30,  # Move to cheaper storage after 30 days
                        "matches_prefix": ["predictions/", "data/processed/"]
                    }
                }
            ]
        },
        "cors": [
            {
                "origin": ["*"],
                "method": ["GET"],
                "responseHeader": ["Content-Type"],
                "maxAgeSeconds": 3600
            }
        ]
    }

def create_requirements_txt() -> str:
    """Create minimal requirements.txt for Cloud Function"""
    
    return """# Minimal requirements for Cloud Function deployment
google-cloud-storage==2.10.0
pandas==2.0.3
numpy==1.24.3
TA-Lib==0.4.26
requests==2.31.0
"""

def create_main_py() -> str:
    """Create main.py for Cloud Function deployment"""
    
    return """# main.py - Cloud Function entry point
from gcs_breakout_system import cloud_function_entry_point

def breakout_predictor(request):
    \"\"\"Cloud Function entry point\"\"\"
    return cloud_function_entry_point(request)
"""

def create_deployment_script() -> str:
    """Create deployment script"""
    
    return """#!/bin/bash
# deploy.sh - Deploy the 40%+ Breakout Prediction System to Google Cloud

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying 40%+ Breakout Prediction System${NC}"

# Check if required environment variables are set
required_vars=("GCS_BUCKET_NAME" "TWELVEDATA_API_KEY" "ALPHAVANTAGE_API_KEY" "PROJECT_ID")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}❌ Error: $var environment variable is not set${NC}"
        exit 1
    fi
done

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Bucket Name: $GCS_BUCKET_NAME"
echo "  Region: us-central1"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable storage-api.googleapis.com

# Create GCS bucket
echo -e "${YELLOW}🪣 Creating GCS bucket...${NC}"
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$GCS_BUCKET_NAME || echo "Bucket may already exist"

# Set bucket permissions for web access
gsutil iam ch allUsers:objectViewer gs://$GCS_BUCKET_NAME

# Create bucket structure
echo -e "${YELLOW}📁 Creating bucket structure...${NC}"
echo "" | gsutil cp - gs://$GCS_BUCKET_NAME/data/raw/.keep
echo "" | gsutil cp - gs://$GCS_BUCKET_NAME/data/processed/.keep
echo "" | gsutil cp - gs://$GCS_BUCKET_NAME/predictions/.keep
echo "" | gsutil cp - gs://$GCS_BUCKET_NAME/web/.keep
echo "" | gsutil cp - gs://$GCS_BUCKET_NAME/web/tickers/.keep

# Deploy Cloud Function
echo -e "${YELLOW}☁️ Deploying Cloud Function...${NC}"
gcloud functions deploy breakout-predictor \\
    --runtime python311 \\
    --trigger-http \\
    --allow-unauthenticated \\
    --memory 512MB \\
    --timeout 540s \\
    --region us-central1 \\
    --set-env-vars GCS_BUCKET_NAME=$GCS_BUCKET_NAME,TWELVEDATA_API_KEY=$TWELVEDATA_API_KEY,ALPHAVANTAGE_API_KEY=$ALPHAVANTAGE_API_KEY

# Get the Cloud Function URL
FUNCTION_URL=$(gcloud functions describe breakout-predictor --region=us-central1 --format="value(httpsTrigger.url)")

# Create Cloud Scheduler job
echo -e "${YELLOW}⏰ Creating daily scheduler...${NC}"
gcloud scheduler jobs create http daily-breakout-scan \\
    --schedule="0 18 * * 1-5" \\
    --timezone="America/New_York" \\
    --uri="$FUNCTION_URL" \\
    --http-method=POST \\
    --headers="Content-Type=application/json" \\
    --message-body='{"tickers":["PLTR","SOFI","HOOD","AFRM","UPST","ROKU","SNAP","CRSR","DKNG","COIN"]}' \\
    --location=us-central1 || echo "Scheduler job may already exist"

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}🌐 Your system is now live at:${NC}"
echo "  Cloud Function: $FUNCTION_URL"
echo "  Web Dashboard: https://storage.googleapis.com/$GCS_BUCKET_NAME/web/index.html"
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "  1. Test the function: curl -X POST $FUNCTION_URL"
echo "  2. View predictions: https://storage.googleapis.com/$GCS_BUCKET_NAME/web/index.html"
echo "  3. Monitor costs: https://console.cloud.google.com/billing"
echo ""
echo -e "${GREEN}💰 Estimated monthly cost: $2-5 USD${NC}"
"""

def create_environment_template() -> str:
    """Create environment variable template"""
    
    return """# .env.cloud - Environment variables for Google Cloud deployment
# Copy this file and fill in your actual values

# Google Cloud Project
PROJECT_ID=your-gcp-project-id

# Storage bucket (must be globally unique)
GCS_BUCKET_NAME=your-unique-bucket-name-breakout-predictor

# API Keys (get free keys from respective services)
TWELVEDATA_API_KEY=your_twelvedata_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key

# Optional: Additional configuration
CLOUD_FUNCTION_REGION=us-central1
SCHEDULER_TIMEZONE=America/New_York
"""

def create_cost_estimation() -> Dict[str, Any]:
    """Create cost estimation for the cloud deployment"""
    
    return {
        "monthly_costs": {
            "cloud_function": {
                "invocations": "~22 per month (daily)",
                "execution_time": "~30 seconds average",
                "cost": "$0.10 - $0.50"
            },
            "cloud_storage": {
                "storage": "~100MB data",
                "requests": "~1000 requests/month",
                "cost": "$0.50 - $1.00"
            },
            "cloud_scheduler": {
                "jobs": "1 daily job",
                "cost": "$0.10"
            },
            "data_apis": {
                "twelvedata": "Free tier (800 requests/day)",
                "alphavantage": "Free tier (500 requests/day)",
                "cost": "$0.00"
            },
            "total_estimated": "$0.70 - $1.60 per month"
        },
        "cost_optimization_tips": [
            "Use free tiers of data APIs (sufficient for small watchlists)",
            "Set lifecycle policies to delete old data automatically",
            "Use us-central1 region for lowest costs",
            "Monitor function execution time and optimize",
            "Use Cloud Storage for hosting instead of App Engine/Compute Engine"
        ]
    }

# Create all deployment files
def generate_deployment_files():
    """Generate all files needed for cloud deployment"""
    
    files = {
        "requirements.txt": create_requirements_txt(),
        "main.py": create_main_py(),
        "deploy.sh": create_deployment_script(),
        ".env.cloud.example": create_environment_template(),
        "cloud_config.json": json.dumps({
            "cloud_function": create_cloud_function_config(),
            "scheduler": create_cloud_scheduler_config(),
            "storage": create_gcs_bucket_config(),
            "cost_estimation": create_cost_estimation()
        }, indent=2)
    }
    
    return files

if __name__ == "__main__":
    print("🚀 Generating Google Cloud deployment files...")
    
    files = generate_deployment_files()
    
    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created {filename}")
    
    print("\\n🎉 Deployment files ready!")
    print("\\n📋 Next steps:")
    print("1. Set up your environment variables in .env.cloud")
    print("2. Run: chmod +x deploy.sh")
    print("3. Run: ./deploy.sh")
    print("\\n💰 Estimated cost: $1-2/month")