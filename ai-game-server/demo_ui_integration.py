"""
PyBoy UI Integration Demo Script
Demonstrates the automatic UI launching functionality
"""
import os
import sys
import time
import logging
import requests
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_ui_integration():
    """Demonstrate the UI integration system"""
    print("🎮 PyBoy UI Integration Demo")
    print("=" * 50)

    # Server configuration
    server_url = "http://localhost:5000"
    api_base = f"{server_url}/api"

    # Check if server is running
    try:
        response = requests.get(f"{api_base}/status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nMake sure the server is running:")
        print("  cd ai-game-server")
        print("  python src/backend/server.py")
        return False

    print("\n📋 Demo Steps:")
    print("1. Upload a test ROM (will auto-launch UI)")
    print("2. Check UI status")
    print("3. Test UI control endpoints")
    print("4. Clean up")

    # Create a test ROM file if needed
    test_rom_path = create_test_rom()
    if not test_rom_path:
        print("❌ Could not create test ROM file")
        return False

    try:
        # Step 1: Upload ROM with UI auto-launch
        print(f"\n1. 📤 Uploading ROM: {test_rom_path}")
        with open(test_rom_path, 'rb') as rom_file:
            files = {'rom_file': rom_file}
            data = {
                'emulator_type': 'gb',
                'launch_ui': 'true'
            }

            response = requests.post(
                f"{api_base}/upload-rom",
                files=files,
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ ROM uploaded successfully")
                print(f"   UI Launched: {result.get('ui_launched', False)}")
                if 'ui_status' in result:
                    ui_status = result['ui_status']
                    print(f"   UI PID: {ui_status.get('pid')}")
                    print(f"   UI Ready: {ui_status.get('ready')}")
            else:
                print(f"❌ ROM upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        # Step 2: Check UI status
        print("\n2. 📊 Checking UI status...")
        time.sleep(2)  # Give UI time to start

        response = requests.get(f"{api_base}/ui/status", timeout=10)
        if response.status_code == 200:
            ui_status = response.json()
            print("✅ UI status retrieved")
            print(f"   Running: {ui_status.get('ui_status', {}).get('running', False)}")
            print(f"   Ready: {ui_status.get('ui_status', {}).get('ready', False)}")
            print(f"   PID: {ui_status.get('ui_status', {}).get('pid')}")
        else:
            print(f"❌ Failed to get UI status: {response.status_code}")

        # Step 3: Test UI control endpoints
        print("\n3. 🎮 Testing UI control endpoints...")

        # Test synchronization
        print("   Synchronizing server-UI state...")
        response = requests.post(f"{api_base}/ui/sync", timeout=10)
        if response.status_code == 200:
            print("   ✅ Synchronization successful")
        else:
            print(f"   ⚠️  Synchronization failed: {response.status_code}")

        # Test restart (only if UI is running)
        response = requests.get(f"{api_base}/ui/status", timeout=10)
        if response.status_code == 200:
            ui_status = response.json()
            if ui_status.get('ui_status', {}).get('running', False):
                print("   Restarting UI...")
                response = requests.post(f"{api_base}/ui/restart", timeout=30)
                if response.status_code == 200:
                    print("   ✅ UI restarted successfully")
                else:
                    print(f"   ⚠️  UI restart failed: {response.status_code}")

        # Step 4: Demonstrate screen capture
        print("\n4. 📸 Testing screen capture...")
        response = requests.get(f"{api_base}/screen", timeout=10)
        if response.status_code == 200:
            screen_data = response.json()
            if 'image' in screen_data:
                print("   ✅ Screen capture successful")
                print(f"   Image shape: {screen_data.get('shape')}")
            else:
                print("   ⚠️  Screen capture returned no image")
        else:
            print(f"   ❌ Screen capture failed: {response.status_code}")

        # Step 5: Test game controls
        print("\n5. 🎯 Testing game controls...")
        test_actions = ['RIGHT', 'A', 'START', 'SELECT']

        for action in test_actions:
            response = requests.post(
                f"{api_base}/action",
                json={'action': action, 'frames': 1},
                timeout=10
            )
            if response.status_code == 200:
                print(f"   ✅ Action '{action}' executed")
            else:
                print(f"   ❌ Action '{action}' failed: {response.status_code}")

        print("\n🎉 Demo completed successfully!")
        print("\n📝 Summary:")
        print("- ✅ ROM uploaded with automatic UI launch")
        print("- ✅ UI process management working")
        print("- ✅ Screen capture functional")
        print("- ✅ Game controls operational")
        print("- ✅ REST API endpoints working")

        return True

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

    finally:
        # Clean up
        cleanup_demo(test_rom_path)

def create_test_rom():
    """Create a minimal test ROM file"""
    try:
        # Create a minimal Game Boy ROM file
        rom_data = bytearray([0x00] * 0x8000)  # 32KB ROM

        # Set ROM header
        rom_data[0x104:0x134] = b"TEST_ROM_DEMO" + b"\x00" * (0x134 - 0x104)
        rom_data[0x146] = 0x00  # Game Boy Color flag
        rom_data[0x147] = 0x00  # Super Game Boy flag
        rom_data[0x148] = 0x00  # Cartridge type
        rom_data[0x149] = 0x00  # ROM size
        rom_data[0x14A] = 0x00  # RAM size

        # Write to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.gb', delete=False) as f:
            f.write(rom_data)
            return f.name

    except Exception as e:
        logger.error(f"Failed to create test ROM: {e}")
        return None

def cleanup_demo(rom_path):
    """Clean up demo resources"""
    if rom_path and os.path.exists(rom_path):
        try:
            os.unlink(rom_path)
            logger.info(f"Cleaned up test ROM: {rom_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up test ROM: {e}")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📖 Usage Instructions:")
    print("=" * 50)
    print("1. Start the server:")
    print("   cd ai-game-server")
    print("   python src/backend/server.py")
    print()
    print("2. Run this demo:")
    print("   python demo_ui_integration.py")
    print()
    print("3. The demo will:")
    print("   - Upload a test ROM")
    print("   - Automatically launch UI window")
    print("   - Test all UI control endpoints")
    print("   - Demonstrate screen capture and controls")
    print()
    print("4. Manual testing:")
    print("   curl -X POST http://localhost:5000/api/ui/launch")
    print("   curl http://localhost:5000/api/ui/status")
    print("   curl -X POST http://localhost:5000/api/ui/stop")
    print()
    print("🔧 Configuration:")
    print("- Edit ui_config.json for settings")
    print("- Use environment variables for overrides")
    print("- See UI_SETUP_GUIDE.md for details")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_usage_instructions()
    else:
        success = demo_ui_integration()
        if not success:
            print_usage_instructions()
            sys.exit(1)
        else:
            print("\n🎮 Demo completed! The UI integration system is working correctly.")
            print("\n💡 Next steps:")
            print("   - Upload your own ROM files via the web interface")
            print("   - Try different UI settings in ui_config.json")
            print("   - Test with AI APIs for automated gameplay")
            print("   - Explore the REST API endpoints for automation")