#!/usr/bin/env python3
"""
Test script for AI provider functionality
"""
import os
import sys
import logging
import json
from unittest.mock import Mock, patch
import requests
import numpy as np
from PIL import Image
import io

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.ai_apis.ai_provider_manager import AIProviderManager, ProviderStatus
from backend.ai_apis.openai_compatible import OpenAICompatibleConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a test game screen image"""
    # Create a simple test image (144x160 like Game Boy)
    img_array = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)

    # Add some pattern to make it look like a game screen
    img_array[50:100, 30:130] = [200, 180, 140]  # Background color
    img_array[70:80, 60:100] = [255, 0, 0]       # Red rectangle
    img_array[85:95, 60:100] = [0, 255, 0]       # Green rectangle

    # Convert to bytes
    img = Image.fromarray(img_array)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=85)
    img_bytes = img_buffer.getvalue()

    return img_bytes

def test_provider_manager():
    """Test the AI provider manager"""
    logger.info("Testing AI Provider Manager...")

    # Create provider manager
    manager = AIProviderManager()

    # Check provider status
    status = manager.get_provider_status()
    logger.info(f"Provider status: {json.dumps(status, indent=2)}")

    # Check available providers
    available = manager.get_available_providers()
    logger.info(f"Available providers: {available}")

    return manager

def test_openai_compatible():
    """Test OpenAI-compatible connector"""
    logger.info("Testing OpenAI-compatible connector...")

    # Test with different configurations
    test_cases = [
        {
            'name': 'Local LM Studio',
            'api_key': 'not-needed',
            'base_url': 'http://localhost:1234/v1',
            'model': 'local-model'
        },
        {
            'name': 'OpenAI',
            'api_key': 'test-key',
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4o'
        },
        {
            'name': 'Auto-detect',
            'api_key': None,
            'base_url': None,
            'model': None
        }
    ]

    for case in test_cases:
        logger.info(f"Testing {case['name']}...")
        try:
            if case['api_key'] or case['base_url']:
                connector = OpenAICompatibleConnector(
                    api_key=case['api_key'] or 'not-needed',
                    base_url=case['base_url'],
                    model=case['model']
                )
                logger.info(f"✓ Created connector for {case['name']}")
                logger.info(f"  Base URL: {connector.base_url}")
                logger.info(f"  Model: {connector.model}")
                logger.info(f"  Client initialized: {connector.client is not None}")
            else:
                logger.info(f"✓ Auto-detection would be used for {case['name']}")
        except Exception as e:
            logger.error(f"✗ Failed to create connector for {case['name']}: {e}")

def test_action_generation():
    """Test action generation with mock data"""
    logger.info("Testing action generation...")

    manager = AIProviderManager()
    test_image = create_test_image()
    test_goal = "Navigate through the game world"
    test_history = ["UP", "RIGHT", "A"]

    # Test with fallback
    action, provider = manager.get_next_action(test_image, test_goal, test_history)
    logger.info(f"Action: {action}, Provider: {provider}")

    # Test with specific provider
    if manager.get_available_providers():
        first_provider = manager.get_available_providers()[0]
        action, provider = manager.get_next_action(test_image, test_goal, test_history, first_provider)
        logger.info(f"Specific provider - Action: {action}, Provider: {provider}")

def test_chat_functionality():
    """Test chat functionality"""
    logger.info("Testing chat functionality...")

    manager = AIProviderManager()
    test_image = create_test_image()
    test_message = "What do you see on the screen?"
    test_context = {
        "current_goal": "Explore the game world",
        "action_history": ["UP", "RIGHT", "A", "B"],
        "game_type": "GB"
    }

    # Test with fallback
    response, provider = manager.chat_with_ai(test_message, test_image, test_context)
    logger.info(f"Chat response: {response[:100]}...")
    logger.info(f"Provider: {provider}")

def test_environment_variables():
    """Test environment variable configuration"""
    logger.info("Testing environment variable configuration...")

    # Set test environment variables
    test_env = {
        'GEMINI_API_KEY': 'test-gemini-key',
        'OPENROUTER_API_KEY': 'test-openrouter-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'OPENAI_ENDPOINT': 'http://localhost:1234/v1',
        'OPENAI_MODEL': 'test-model',
        'AI_TIMEOUT': '120',
        'AI_MAX_RETRIES': '5'
    }

    # Store original values
    original_env = {}
    for key in test_env:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env[key]

    try:
        # Test that environment variables are used
        manager = AIProviderManager()

        # Check if providers are initialized with test keys
        status = manager.get_provider_status()
        logger.info(f"Provider status with test env vars: {json.dumps(status, indent=2)}")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    logger.info("Testing error handling...")

    manager = AIProviderManager()
    test_image = create_test_image()
    test_goal = "Test error handling"
    test_history = []

    # Test with no providers available (temporarily disable all)
    original_providers = manager.providers.copy()
    try:
        # Simulate no available providers
        manager.providers = {}

        action, provider = manager.get_next_action(test_image, test_goal, test_history)
        logger.info(f"Default action when no providers: {action}")

        response, provider = manager.chat_with_ai("Test message", test_image, {})
        logger.info(f"Default chat response when no providers: {response}")

    finally:
        manager.providers = original_providers

def main():
    """Run all tests"""
    logger.info("Starting AI provider tests...")

    try:
        # Test provider manager
        manager = test_provider_manager()

        # Test OpenAI-compatible connector
        test_openai_compatible()

        # Test action generation
        test_action_generation()

        # Test chat functionality
        test_chat_functionality()

        # Test environment variables
        test_environment_variables()

        # Test error handling
        test_error_handling()

        logger.info("✓ All tests completed successfully!")

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()