#!/usr/bin/env python3
"""
Demonstration script showing AI features working with local providers
"""
import os
import sys
import json
import time
import requests
import base64
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.ai_apis.ai_provider_manager import AIProviderManager
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_game_state():
    """Create a demo game state (would normally come from emulator)"""
    # Create a simple game-like image
    import numpy as np
    from PIL import Image
    import io

    # Create a 144x160 image (Game Boy size)
    img_array = np.zeros((144, 160, 3), dtype=np.uint8)

    # Add some game-like elements
    # Background (grass)
    img_array[:, :] = [120, 180, 120]

    # Player character (red square)
    img_array[60:80, 70:90] = [255, 0, 0]

    # Obstacle (tree)
    img_array[40:100, 120:140] = [0, 100, 0]

    # Goal area (yellow)
    img_array[10:30, 10:30] = [255, 255, 0]

    # Convert to bytes
    img = Image.fromarray(img_array)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=85)
    img_bytes = img_buffer.getvalue()

    return img_bytes

def test_ai_action_pipeline():
    """Test the complete AI action pipeline"""
    print("\nTesting AI Action Pipeline")
    print("=" * 50)

    # Initialize provider manager
    manager = AIProviderManager()
    available_providers = manager.get_available_providers()

    if not available_providers:
        print("No AI providers available. Please configure providers first.")
        return False

    print(f"Found {len(available_providers)} available providers: {', '.join(available_providers)}")

    # Create test game state
    game_image = create_demo_game_state()
    test_goal = "Navigate to the yellow goal area while avoiding the green obstacle"
    action_history = ["UP", "RIGHT", "A"]

    print(f"\nGame Goal: {test_goal}")
    print(f"Action History: {action_history}")

    # Test with each available provider
    for provider_name in available_providers:
        print(f"\nTesting with {provider_name}...")
        try:
            # Get AI action
            action, actual_provider = manager.get_next_action(
                game_image, test_goal, action_history, provider_name
            )

            print(f"   AI Action: {action}")
            print(f"   Provider Used: {actual_provider}")

            # Test chat functionality
            chat_response, chat_provider = manager.chat_with_ai(
                "What do you see on the screen and what should I do next?",
                game_image,
                {
                    "current_goal": test_goal,
                    "action_history": action_history,
                    "game_type": "GB"
                },
                provider_name
            )

            print(f"   AI Chat: {chat_response[:100]}...")
            print(f"   Chat Provider: {chat_provider}")

        except Exception as e:
            print(f"   Error with {provider_name}: {e}")

    print("\nAI action pipeline test completed!")
    return True

def test_server_endpoints():
    """Test server endpoints (requires server to be running separately)"""
    print("\n🌐 Testing Server Endpoints")
    print("=" * 50)

    print("💡 Note: This test requires the server to be running on localhost:5000")
    print("   Run 'python start_server.py' in another terminal first")

    try:
        # Test health check
        response = requests.get('http://localhost:5000/health', timeout=5)
        print(f"✅ Health Check: {response.status_code}")

        # Test provider status
        response = requests.get('http://localhost:5000/api/providers/status', timeout=5)
        provider_status = response.json()
        print(f"✅ Provider Status: {len([p for p in provider_status.values() if p['available']])} available")

        # Test AI action endpoint
        game_image = create_demo_game_state()
        image_b64 = base64.b64encode(game_image).decode('utf-8')

        action_data = {
            "goal": "Navigate through the game",
            "image": image_b64
        }

        response = requests.post(
            'http://localhost:5000/api/ai-action',
            json=action_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ AI Action: {result.get('action', 'Unknown')}")
            print(f"✅ Provider Used: {result.get('provider_used', 'Unknown')}")
        else:
            print(f"⚠️  AI Action failed: {response.status_code}")

        # Test chat endpoint
        chat_data = {
            "message": "What do you see on the screen?",
            "image": image_b64
        }

        response = requests.post(
            'http://localhost:5000/api/chat',
            json=chat_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat Response: {result.get('response', 'No response')[:50]}...")
            print(f"✅ Chat Provider: {result.get('provider_used', 'Unknown')}")
        else:
            print(f"⚠️  Chat failed: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start with 'python start_server.py'")
        return False
    except Exception as e:
        print(f"❌ Server endpoint test failed: {e}")
        return False

    print("\n✅ Server endpoint test completed!")
    return True

def demo_provider_fallback():
    """Demonstrate provider fallback functionality"""
    print("\n🔄 Demonstrating Provider Fallback")
    print("=" * 50)

    manager = AIProviderManager()
    game_image = create_demo_game_state()
    test_goal = "Test fallback behavior"
    action_history = []

    # Test with a provider that might not exist
    print("🧪 Testing with non-existent provider...")
    action, provider = manager.get_next_action(game_image, test_goal, action_history, "nonexistent_provider")
    print(f"   🎮 Action: {action}")
    print(f"   🏷️  Provider Used: {provider}")

    # Test with available providers
    available_providers = manager.get_available_providers()
    if available_providers:
        print(f"\n🧪 Testing with available provider: {available_providers[0]}")
        action, provider = manager.get_next_action(game_image, test_goal, action_history, available_providers[0])
        print(f"   🎮 Action: {action}")
        print(f"   🏷️  Provider Used: {provider}")

    print("\n✅ Provider fallback demo completed!")

def show_configuration_status():
    """Show current configuration status"""
    print("\n⚙️  Configuration Status")
    print("=" * 50)

    # Environment variables
    env_vars = [
        'GEMINI_API_KEY',
        'OPENROUTER_API_KEY',
        'OPENAI_API_KEY',
        'OPENAI_ENDPOINT',
        'NVIDIA_API_KEY',
        'AI_TIMEOUT',
        'AI_MAX_RETRIES'
    ]

    print("📋 Environment Variables:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var.endswith('_API_KEY') and value != 'Not set':
            value = '***' + value[-4:] if len(value) > 4 else '***'
        print(f"   {var}: {value}")

    # Provider status
    manager = AIProviderManager()
    status = manager.get_provider_status()

    print(f"\n🤖 AI Provider Status:")
    for provider_name, info in status.items():
        status_text = "✅ Available" if info['available'] else "❌ Unavailable"
        print(f"   {provider_name}: {status_text}")
        if info['error']:
            print(f"      Error: {info['error']}")

def main():
    """Main demo function"""
    print("AI Game Server Feature Demo")
    print("=" * 60)

    # Show configuration
    show_configuration_status()

    # Test AI action pipeline
    test_ai_action_pipeline()

    # Test server endpoints
    test_server_endpoints()

    # Demo provider fallback
    demo_provider_fallback()

    print("\nDemo completed!")
    print("\nNext steps:")
    print("   1. Configure your AI providers in .env file")
    print("   2. Start a local AI provider (LM Studio, Ollama)")
    print("   3. Run: python start_server.py")
    print("   4. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()