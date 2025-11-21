#!/usr/bin/env python3
"""
Simple demonstration of AI features without emojis
"""
import os
import sys
import json
import time
import numpy as np
from PIL import Image
import io

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.ai_apis.ai_provider_manager import AIProviderManager

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

def main():
    """Main demo function"""
    print("AI Game Server - Simple Feature Demo")
    print("=" * 50)

    # Initialize provider manager
    manager = AIProviderManager()

    # Check available providers
    available_providers = manager.get_available_providers()
    print(f"Available providers: {available_providers}")

    if not available_providers:
        print("No AI providers available. Please check your configuration.")
        print("See AI_PROVIDER_SETUP.md for setup instructions.")
        return

    # Create test data
    test_image = create_test_image()
    test_goal = "Navigate through the game world"
    test_history = ["UP", "RIGHT", "A"]

    print(f"\nTest goal: {test_goal}")
    print(f"Action history: {test_history}")

    # Test with each provider
    for provider_name in available_providers:
        print(f"\n--- Testing {provider_name} ---")

        try:
            # Test action generation
            action, actual_provider = manager.get_next_action(
                test_image, test_goal, test_history, provider_name
            )
            print(f"Action: {action}")
            print(f"Provider used: {actual_provider}")

            # Test chat functionality
            chat_response, chat_provider = manager.chat_with_ai(
                "What do you see on the screen?",
                test_image,
                {
                    "current_goal": test_goal,
                    "action_history": test_history,
                    "game_type": "GB"
                },
                provider_name
            )
            print(f"Chat response: {chat_response[:100]}...")
            print(f"Chat provider: {chat_provider}")

        except Exception as e:
            print(f"Error with {provider_name}: {e}")

    # Test fallback behavior
    print("\n--- Testing Fallback Behavior ---")
    try:
        action, provider = manager.get_next_action(
            test_image, test_goal, test_history, "nonexistent_provider"
        )
        print(f"Fallback action: {action}")
        print(f"Fallback provider: {provider}")
    except Exception as e:
        print(f"Fallback error: {e}")

    # Show provider status
    print("\n--- Provider Status ---")
    status = manager.get_provider_status()
    for provider_name, info in status.items():
        status_text = "Available" if info['available'] else "Unavailable"
        print(f"{provider_name}: {status_text}")
        if info['error']:
            print(f"  Error: {info['error']}")

    print("\nDemo completed successfully!")
    print("\nNext steps:")
    print("1. Configure your AI providers in .env file")
    print("2. Run: python start_server.py")
    print("3. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()