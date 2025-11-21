#!/usr/bin/env python3
"""
Test script to verify AI API endpoints are working correctly
"""
import json
import requests
import time
import os
from typing import Dict, Any

# Server configuration
SERVER_URL = "http://localhost:5000"
API_ENDPOINTS = {
    "health": "/health",
    "status": "/api/status",
    "ai_action": "/api/ai-action",
    "action": "/api/action",
    "screen": "/api/screen",
    "chat": "/api/chat"
}

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test a specific API endpoint"""
    url = f"{SERVER_URL}{endpoint}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method.upper() == "OPTIONS":
            response = requests.options(url, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}

        result = {
            "status_code": response.status_code,
            "url": url,
            "method": method,
            "success": response.status_code < 400,
            "headers": dict(response.headers),
            "cors_headers": {
                "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
                "access-control-allow-headers": response.headers.get("access-control-allow-headers"),
            }
        }

        # Try to parse JSON response
        try:
            result["data"] = response.json()
        except:
            result["data"] = response.text[:500]  # First 500 characters

        return result

    except requests.exceptions.ConnectionError:
        return {
            "error": f"Connection failed - server not running at {SERVER_URL}",
            "url": url,
            "method": method,
            "success": False
        }
    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out",
            "url": url,
            "method": method,
            "success": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "url": url,
            "method": method,
            "success": False
        }

def main():
    """Main test function"""
    print("🔍 Testing AI Game Server API Endpoints")
    print("=" * 50)

    # Test basic endpoints first
    print("\n1. Testing Health Endpoint...")
    health_result = test_endpoint(API_ENDPOINTS["health"])
    print(f"   Status: {health_result.get('status_code', 'N/A')}")
    if health_result.get('success'):
        print("   ✅ Health check passed")
    else:
        print("   ❌ Health check failed")
        print(f"   Error: {health_result.get('error', 'Unknown error')}")
        return

    # Test status endpoint
    print("\n2. Testing Status Endpoint...")
    status_result = test_endpoint(API_ENDPOINTS["status"])
    print(f"   Status: {status_result.get('status_code', 'N/A')}")
    if status_result.get('success'):
        print("   ✅ Status endpoint working")
        game_state = status_result.get('data', {})
        print(f"   ROM loaded: {game_state.get('rom_loaded', False)}")
        print(f"   Active emulator: {game_state.get('active_emulator', 'None')}")
    else:
        print("   ❌ Status endpoint failed")

    # Test CORS preflight for AI action endpoint
    print("\n3. Testing CORS Preflight for AI Action...")
    cors_result = test_endpoint(API_ENDPOINTS["ai_action"], "OPTIONS")
    print(f"   Status: {cors_result.get('status_code', 'N/A')}")
    if cors_result.get('success'):
        cors_headers = cors_result.get('cors_headers', {})
        print(f"   CORS Origin: {cors_headers.get('access-control-allow-origin', 'None')}")
        print(f"   CORS Methods: {cors_headers.get('access-control-allow-methods', 'None')}")
        print("   ✅ CORS preflight working")
    else:
        print("   ❌ CORS preflight failed")

    # Test AI action endpoint (without ROM - should fail gracefully)
    print("\n4. Testing AI Action Endpoint (without ROM)...")
    ai_data = {
        "api_name": "gemini",
        "api_key": "test_key",
        "goal": "Test goal",
        "api_endpoint": None
    }
    ai_result = test_endpoint(API_ENDPOINTS["ai_action"], "POST", ai_data)
    print(f"   Status: {ai_result.get('status_code', 'N/A')}")
    if ai_result.get('success'):
        print("   ✅ AI action endpoint reachable")
    else:
        print("   ❌ AI action endpoint failed")
        print(f"   Error: {ai_result.get('error', 'Unknown error')}")

    # Test action endpoint (without ROM - should fail gracefully)
    print("\n5. Testing Action Endpoint (without ROM)...")
    action_data = {
        "action": "UP",
        "frames": 1
    }
    action_result = test_endpoint(API_ENDPOINTS["action"], "POST", action_data)
    print(f"   Status: {action_result.get('status_code', 'N/A')}")
    if action_result.get('success'):
        print("   ✅ Action endpoint reachable")
    else:
        print("   ❌ Action endpoint failed")
        print(f"   Error: {action_result.get('error', 'Unknown error')}")

    # Test screen endpoint (without ROM - should fail gracefully)
    print("\n6. Testing Screen Endpoint...")
    screen_result = test_endpoint(API_ENDPOINTS["screen"])
    print(f"   Status: {screen_result.get('status_code', 'N/A')}")
    if screen_result.get('success'):
        print("   ✅ Screen endpoint reachable")
    else:
        print("   ❌ Screen endpoint failed")
        print(f"   Error: {screen_result.get('error', 'Unknown error')}")

    # Check environment variables
    print("\n7. Checking Environment Variables...")
    api_keys = {
        "GEMINI_API_KEY": os.environ.get('GEMINI_API_KEY'),
        "OPENROUTER_API_KEY": os.environ.get('OPENROUTER_API_KEY'),
        "NVIDIA_API_KEY": os.environ.get('NVIDIA_API_KEY'),
        "OPENAI_API_KEY": os.environ.get('OPENAI_API_KEY')
    }

    configured_apis = [key for key, value in api_keys.items() if value and value.strip()]
    print(f"   Configured APIs: {len(configured_apis)}/4")
    for api in configured_apis:
        print(f"   ✅ {api}")

    missing_apis = [key for key, value in api_keys.items() if not value or not value.strip()]
    for api in missing_apis:
        print(f"   ❌ {api} - Not configured")

    print("\n" + "=" * 50)
    print("🏁 API Endpoint Testing Complete")

    if len(configured_apis) == 0:
        print("\n⚠️  IMPORTANT: No API keys are configured!")
        print("   AI features will not work until you set up API keys.")
        print("   Run 'start_server_with_env.bat' and replace placeholder keys.")

    return health_result.get('success', False)

if __name__ == "__main__":
    main()