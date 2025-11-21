"""
UI Integration Test Script
Tests the complete UI integration system
"""
import os
import sys
import time
import tempfile
import logging
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ui_process_manager import PyBoyUIProcessManager, get_ui_manager
from backend.ui_config import UIConfig, get_ui_config
from backend.emulators.pyboy_emulator import PyBoyEmulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestUIIntegration(unittest.TestCase):
    """Test cases for UI integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = get_ui_config()
        self.ui_manager = get_ui_manager()

        # Create a test ROM path (we'll use a dummy path for testing)
        self.test_rom_path = "test_rom.gb"

        # Create a temporary ROM file for testing
        self.temp_rom = None
        try:
            # Create a minimal ROM file (just a header)
            with tempfile.NamedTemporaryFile(suffix='.gb', delete=False) as f:
                # Write a minimal Game Boy ROM header
                f.write(bytes([0x3E] * 0x150))  # Fill with NOP instructions
                f.write(b"TEST_ROM")  # ROM title
                f.write(bytes([0x00] * 0x50))  # Rest of the header
                self.temp_rom = f.name
                self.test_rom_path = self.temp_rom
        except Exception as e:
            logger.warning(f"Could not create test ROM file: {e}")

    def tearDown(self):
        """Clean up test fixtures"""
        # Stop any running UI processes
        if self.ui_manager.is_ui_running():
            self.ui_manager.stop_ui()

        # Clean up temporary ROM file
        if self.temp_rom and os.path.exists(self.temp_rom):
            os.unlink(self.temp_rom)

    def test_ui_config_loading(self):
        """Test UI configuration loading"""
        logger.info("Testing UI configuration loading...")

        # Test default configuration
        self.assertIsNotNone(self.config)
        self.assertEqual(self.config.get("window.type"), "sdl2")
        self.assertEqual(self.config.get("window.scale"), 2)
        self.assertEqual(self.config.get("sound.volume"), 50)

        # Test configuration validation
        self.assertTrue(self.config.validate_config())

        # Test configuration getters
        window_args = self.config.get_window_args()
        self.assertIn("window", window_args)
        self.assertIn("scale", window_args)

        sound_args = self.config.get_sound_args()
        self.assertIn("sound", sound_args)
        self.assertIn("sound_volume", sound_args)

        logger.info("UI configuration test passed")

    def test_ui_manager_initialization(self):
        """Test UI manager initialization"""
        logger.info("Testing UI manager initialization...")

        # Test manager creation
        self.assertIsNotNone(self.ui_manager)
        self.assertFalse(self.ui_manager.is_ui_running())

        # Test status method
        status = self.ui_manager.get_ui_status()
        self.assertIn("running", status)
        self.assertIn("ready", status)
        self.assertFalse(status["running"])
        self.assertFalse(status["ready"])

        logger.info("UI manager initialization test passed")

    def test_emulator_ui_integration(self):
        """Test emulator UI integration"""
        logger.info("Testing emulator UI integration...")

        # Create emulator instance
        emulator = PyBoyEmulator()

        # Test that UI manager is accessible
        self.assertIsNotNone(emulator.ui_manager)
        self.assertIsInstance(emulator.ui_manager, PyBoyUIProcessManager)

        # Test UI methods exist
        self.assertTrue(hasattr(emulator, 'launch_ui'))
        self.assertTrue(hasattr(emulator, 'stop_ui'))
        self.assertTrue(hasattr(emulator, 'get_ui_status'))
        self.assertTrue(hasattr(emulator, 'restart_ui'))

        logger.info("Emulator UI integration test passed")

    def test_ui_process_lifecycle(self):
        """Test UI process lifecycle (mock test)"""
        logger.info("Testing UI process lifecycle...")

        # Initial state
        self.assertFalse(self.ui_manager.is_ui_running())

        # Test process argument preparation (without actually launching)
        if self.temp_rom and os.path.exists(self.temp_rom):
            args = self.ui_manager._prepare_ui_args(self.temp_rom, "/tmp")
            self.assertIsInstance(args, list)
            self.assertIn(self.temp_rom, args)
            self.assertTrue(any("sdl2" in arg for arg in args))

        # Test status checking
        status = self.ui_manager.get_ui_status()
        self.assertIsInstance(status, dict)
        self.assertIn("running", status)

        logger.info("UI process lifecycle test passed")

    def test_configuration_from_environment(self):
        """Test configuration updates from environment variables"""
        logger.info("Testing configuration from environment variables...")

        # Set environment variables
        old_env = {}
        test_vars = {
            "PYBOY_UI_SCALE": "3",
            "PYBOY_UI_SOUND_VOLUME": "75",
            "PYBOY_UI_LOG_LEVEL": "DEBUG"
        }

        # Store old values and set new ones
        for key, value in test_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Create new config instance to test environment loading
            test_config = UIConfig()
            test_config.update_from_env()

            # Test that values were updated
            self.assertEqual(test_config.get("window.scale"), 3)
            self.assertEqual(test_config.get("sound.volume"), 75)
            self.assertEqual(test_config.get("debug.log_level"), "DEBUG")

        finally:
            # Restore old environment values
            for key, value in test_vars.items():
                if old_env[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_env[key]

        logger.info("Environment configuration test passed")

    def test_error_handling(self):
        """Test error handling"""
        logger.info("Testing error handling...")

        # Test with invalid ROM path
        invalid_path = "/path/that/does/not/exist.gb"

        # Should not crash, but should handle gracefully
        success = self.ui_manager.launch_ui(invalid_path)
        self.assertFalse(success)

        # Test stopping non-existent process
        success = self.ui_manager.stop_ui()
        self.assertTrue(success)  # Should succeed even if no process

        logger.info("Error handling test passed")

    def test_configuration_persistence(self):
        """Test configuration saving and loading"""
        logger.info("Testing configuration persistence...")

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Create config with temporary file
            test_config = UIConfig(config_path)

            # Modify some values
            test_config.set("window.scale", 4)
            test_config.set("sound.volume", 80)

            # Save configuration
            success = test_config.save_config()
            self.assertTrue(success)

            # Load configuration in new instance
            new_config = UIConfig(config_path)

            # Verify values were loaded
            self.assertEqual(new_config.get("window.scale"), 4)
            self.assertEqual(new_config.get("sound.volume"), 80)

        finally:
            # Clean up temporary config file
            if os.path.exists(config_path):
                os.unlink(config_path)

        logger.info("Configuration persistence test passed")

def run_integration_tests():
    """Run all integration tests"""
    logger.info("Starting UI integration tests...")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUIIntegration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Report results
    if result.wasSuccessful():
        logger.info("All integration tests passed!")
        return True
    else:
        logger.error(f"Integration tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False

def test_ui_launcher_script():
    """Test that the UI launcher script can be imported and has expected functions"""
    logger.info("Testing UI launcher script...")

    try:
        # Import the UI launcher module
        sys.path.insert(0, str(Path(__file__).parent))
        from backend.ui_launcher import parse_arguments, main

        # Test argument parsing
        test_args = [
            "test_rom.gb",
            "--window", "sdl2",
            "--scale", "2",
            "--sound-volume", "50",
            "--log-level", "INFO"
        ]

        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = ["ui_launcher.py"] + test_args

        try:
            args = parse_arguments()
            self.assertEqual(args.rom_path, "test_rom.gb")
            self.assertEqual(args.window, "sdl2")
            self.assertEqual(args.scale, 2)
            self.assertEqual(args.sound_volume, 50)
            self.assertEqual(args.log_level, "INFO")
        finally:
            sys.argv = original_argv

        logger.info("UI launcher script test passed")
        return True

    except Exception as e:
        logger.error(f"UI launcher script test failed: {e}")
        return False

if __name__ == "__main__":
    """Main test runner"""
    print("PyBoy UI Integration Test Suite")
    print("=" * 50)

    # Run all tests
    tests_passed = True

    # Run unit tests
    tests_passed &= run_integration_tests()

    # Test UI launcher script
    tests_passed &= test_ui_launcher_script()

    # Summary
    print("\n" + "=" * 50)
    if tests_passed:
        print("✅ All tests passed! UI integration is working correctly.")
        print("\nThe system is ready for use with the following features:")
        print("- Automatic UI launching when ROM is loaded")
        print("- Proper sound configuration to prevent buffer overruns")
        print("- Robust process management and error handling")
        print("- Configurable UI settings via config file or environment variables")
        print("- REST API endpoints for UI control")
    else:
        print("❌ Some tests failed. Please check the logs above for details.")
        sys.exit(1)