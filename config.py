"""
Configuration for baseline evaluation.
Edit these settings before running.
"""

import os

# ============================================================
# API Configuration
# ============================================================
TINKER_API_KEY = os.environ.get('TINKER_API_KEY', 'your-api-key-here')

# ============================================================
# Model Configuration
# ============================================================
SMALL_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LARGE_MODEL = "deepseek-ai/DeepSeek-V3.1"

# ============================================================
# Dataset Configuration
# ============================================================
DATASETS = {
    'gsm8k': {
        'enabled': True,
        'num_samples': 500,
        'max_tokens': 400,
    },
    'sat': {
        'enabled': True,
        'num_samples': 300,
        'max_tokens': 20,
    }
}

# ============================================================
# Execution Configuration
# ============================================================
CHECKPOINT_INTERVAL = 50  # Save every N samples
MAX_RETRIES = 3  # Retry failed samples
RETRY_DELAY = 5  # Seconds between retries

# ============================================================
# Output Configuration
# ============================================================
RESULTS_DIR = './results'
CHECKPOINT_DIR = './results/checkpoints'
LOGS_DIR = './results/logs'

# ============================================================
# Notification Configuration (optional)
# ============================================================
ENABLE_EMAIL = False  # Set to True if you want email notifications
EMAIL_ADDRESS = 'your-email@example.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
