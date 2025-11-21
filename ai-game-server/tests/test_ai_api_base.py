"""
Unit tests for the AI API base class
"""
import unittest
import sys
import os
import base64

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.ai_api_base import AIAPIConnector


class MockAIAPIConnector(AIAPIConnector):
    """Mock implementation of the AI API connector for testing"""
    
    def get_next_action(self, image_data: bytes, goal: str, history: list) -> str:
        """Mock implementation that always returns 'A'"""
        return 'A'


class TestAIAPIBase(unittest.TestCase):
    
    def test_interface_cannot_be_instantiated(self):
        """Test that the base class cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            connector = AIAPIConnector("test-key")
    
    def test_mock_implementation(self):
        """Test that the mock implementation works"""
        connector = MockAIAPIConnector("test-key")
        
        # Test that the API key is set correctly
        self.assertEqual(connector.api_key, "test-key")
        
        # Test that get_next_action returns the expected value
        result = connector.get_next_action(b"test-image-data", "test-goal", ["A", "B"])
        self.assertEqual(result, "A")
    
    def test_encode_image(self):
        """Test the encode_image method"""
        connector = MockAIAPIConnector("test-key")
        
        # Test with simple byte data
        test_data = b"test image data"
        encoded = connector.encode_image(test_data)
        
        # Check that it's a string
        self.assertIsInstance(encoded, str)
        
        # Check that it can be decoded back to the original data
        decoded = base64.b64decode(encoded)
        self.assertEqual(decoded, test_data)


if __name__ == '__main__':
    unittest.main()