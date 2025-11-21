"""
Test script for AI Game Server
"""
import requests
import json
import base64
import time

# Server URL
SERVER_URL = "http://localhost:5000"

def test_server_status():
    """Test the server status endpoint"""
    print("Testing server status...")
    try:
        response = requests.get(f"{SERVER_URL}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_load_rom(rom_path, emulator_type="gb"):
    """Test loading a ROM"""
    print(f"Testing ROM load: {rom_path}")
    try:
        data = {
            "rom_path": rom_path,
            "emulator_type": emulator_type
        }
        response = requests.post(f"{SERVER_URL}/api/load-rom", json=data)
        print(f"Load ROM: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_screen():
    """Test getting the screen"""
    print("Testing screen capture...")
    try:
        response = requests.get(f"{SERVER_URL}/api/screen")
        print(f"Screen: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Screen shape: {data.get('shape')}")
            # Save the image to a file for verification
            if 'image' in data:
                with open("test_screen.jpg", "wb") as f:
                    f.write(base64.b64decode(data['image']))
                print("Screen saved to test_screen.jpg")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_execute_action(action="A", frames=10):
    """Test executing an action"""
    print(f"Testing action: {action}")
    try:
        data = {
            "action": action,
            "frames": frames
        }
        response = requests.post(f"{SERVER_URL}/api/action", json=data)
        print(f"Action: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_ai_action(goal="Defeat the first gym leader", api_name="gemini"):
    """Test getting an AI action"""
    print(f"Testing AI action with goal: {goal}")
    try:
        data = {
            "goal": goal,
            "api_name": api_name
        }
        response = requests.post(f"{SERVER_URL}/api/ai-action", json=data)
        print(f"AI Action: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Action: {result.get('action')}")
            print(f"History: {result.get('history')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_info():
    """Test getting game info"""
    print("Testing game info...")
    try:
        response = requests.get(f"{SERVER_URL}/api/info")
        print(f"Info: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("AI Game Server Test Script")
    print("=" * 30)
    
    # Test server status
    if not test_server_status():
        print("Server is not running. Please start the server first.")
        return
    
    # Test other endpoints
    print("\n" + "=" * 30)
    
    # Note: The following tests require a running server with ROMs loaded
    # Uncomment and modify as needed for your testing
    
    # Test loading a ROM (you'll need to provide a valid ROM path)
    # test_load_rom("/path/to/your/rom.gb", "gb")
    
    # Test getting screen
    # test_get_screen()
    
    # Test executing an action
    # test_execute_action("A", 10)
    
    # Test AI action
    # test_ai_action("Get out of the house", "gemini")
    
    # Test getting info
    # test_get_info()
    
    print("\nTest script completed.")

if __name__ == "__main__":
    main()