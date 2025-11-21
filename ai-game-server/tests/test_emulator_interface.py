"""
Unit tests for the emulator interface
"""
import unittest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.emulators.emulator_interface import EmulatorInterface


class TestEmulatorInterface(unittest.TestCase):
    
    def test_interface_cannot_be_instantiated(self):
        """Test that the interface cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            emulator = EmulatorInterface()
    
    def test_interface_methods_exist(self):
        """Test that all required methods are defined in the interface"""
        # Get all methods from the interface
        methods = [method for method in dir(EmulatorInterface) if not method.startswith('_')]
        
        # Check that all required methods are present
        required_methods = [
            'load_rom',
            'step',
            'get_screen',
            'get_memory',
            'set_memory',
            'reset',
            'save_state',
            'load_state',
            'get_info'
        ]
        
        for method in required_methods:
            self.assertIn(method, methods)


if __name__ == '__main__':
    unittest.main()