# AI Game Server Configuration

# Server settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Emulator settings
DEFAULT_EMULATOR = "gb"  # gb or gba
SCREEN_CAPTURE_FORMAT = "jpeg"
SCREEN_CAPTURE_QUALITY = 85

# AI API settings
DEFAULT_AI_API = "gemini"
AI_REQUEST_TIMEOUT = 30  # seconds

# Provider-specific settings
AI_TIMEOUT = 60  # seconds
AI_MAX_RETRIES = 3
AI_MODEL = "gpt-4o"  # Default model for OpenAI-compatible providers

# Environment variable names for AI providers
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_ENDPOINT_ENV = "OPENAI_ENDPOINT"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_MODEL_ENV = "NVIDIA_MODEL"

# Local provider settings
LOCAL_PROVIDER_URL = "http://localhost:1234/v1"  # Default for LM Studio
LM_STUDIO_PORT = 1234
LM_STUDIO_URL = "http://localhost:1234/v1"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "{asctime} - {name} - {levelname} - {message}"
LOG_FILE = "ai_game_server.log"

# Security settings
MAX_ROM_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_ROM_EXTENSIONS = [".gb", ".gbc", ".gba"]

# Performance settings
ACTION_HISTORY_LIMIT = 1000
SCREEN_UPDATE_INTERVAL = 0.1  # seconds