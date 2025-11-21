"""
Comprehensive test suite for AI API base class functionality
"""
import unittest
import sys
import os
import base64
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.ai_api_base import AIAPIConnector


class ComprehensiveMockAIAPIConnector(AIAPIConnector):
    """Comprehensive mock implementation of the AI API connector for testing"""

    def __init__(self, api_key: str, should_fail: bool = False, delay: float = 0):
        super().__init__(api_key)
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0
        self.last_request = None

    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        """Mock implementation that can simulate failures and delays"""
        self.call_count += 1
        self.last_request = {
            'image_data': image_data,
            'goal': goal,
            'history': history
        }

        if self.delay > 0:
            time.sleep(self.delay)

        if self.should_fail:
            raise Exception("Simulated API failure")

        # Return different actions based on goal for testing
        if "right" in goal.lower():
            return "RIGHT"
        elif "left" in goal.lower():
            return "LEFT"
        elif "up" in goal.lower():
            return "UP"
        elif "down" in goal.lower():
            return "DOWN"
        elif "start" in goal.lower():
            return "START"
        elif "select" in goal.lower():
            return "SELECT"
        else:
            return "A"

    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
        """Mock chat implementation"""
        self.call_count += 1
        self.last_request = {
            'message': message,
            'image_data': image_data,
            'context': context
        }

        if self.should_fail:
            raise Exception("Simulated chat failure")

        return f"Mock response to: {message}"


class TestAIAPIBaseComprehensive(unittest.TestCase):
    """Comprehensive test suite for AI API base class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-api-key-12345"
        self.valid_connector = ComprehensiveMockAIAPIConnector(self.test_api_key)
        self.failing_connector = ComprehensiveMockAIAPIConnector(self.test_api_key, should_fail=True)

        # Create test image data
        self.test_image_data = b"test-image-data-" + b"x" * 1000

    def test_interface_cannot_be_instantiated_directly(self):
        """Test that the base class cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            connector = AIAPIConnector("test-key")

    def test_api_key_validation_on_init(self):
        """Test API key validation during initialization"""
        # Test with valid key
        connector = ComprehensiveMockAIAPIConnector("valid-key-12345")
        self.assertEqual(connector.api_key, "valid-key-12345")

        # Test with empty key (should not raise exception but should log warning)
        with patch('logging.Logger.warning') as mock_warning:
            connector = ComprehensiveMockAIAPIConnector("")
            mock_warning.assert_called()

        # Test with None key
        with patch('logging.Logger.warning') as mock_warning:
            connector = ComprehensiveMockAIAPIConnector(None)
            mock_warning.assert_called()

        # Test with short key
        with patch('logging.Logger.warning') as mock_warning:
            connector = ComprehensiveMockAIAPIConnector("short")
            mock_warning.assert_called()

    def test_retry_logic_success_on_first_attempt(self):
        """Test retry logic succeeds on first attempt"""
        result = self.valid_connector._retry_with_backoff(
            lambda: "success"
        )
        self.assertEqual(result, "success")

    def test_retry_logic_success_after_failures(self):
        """Test retry logic succeeds after some failures"""
        attempt_count = 0

        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "eventual_success"

        result = self.valid_connector._retry_with_backoff(flaky_function)
        self.assertEqual(result, "eventual_success")
        self.assertEqual(attempt_count, 3)

    def test_retry_logic_exhausted_attempts(self):
        """Test retry logic when all attempts are exhausted"""
        def always_fail_function():
            raise Exception("Always fails")

        with self.assertRaises(Exception):
            self.valid_connector._retry_with_backoff(always_fail_function)

    def test_retry_logic_with_custom_settings(self):
        """Test retry logic with custom retry settings"""
        connector = ComprehensiveMockAIAPIConnector("test-key")
        connector.max_retries = 5
        connector.retry_delay = 0.1

        attempt_count = 0

        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 4:
                raise Exception("Temporary failure")
            return "success"

        result = connector._retry_with_backoff(flaky_function)
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 4)

    def test_encode_image_functionality(self):
        """Test image encoding functionality"""
        # Test with valid image data
        test_data = b"test image data for encoding"
        encoded = self.valid_connector.encode_image(test_data)

        # Verify it's a string
        self.assertIsInstance(encoded, str)

        # Verify it can be decoded back
        decoded = base64.b64decode(encoded)
        self.assertEqual(decoded, test_data)

        # Test with empty data
        empty_encoded = self.valid_connector.encode_image(b"")
        self.assertEqual(base64.b64decode(empty_encoded), b"")

        # Test with binary data
        binary_data = bytes([0, 1, 2, 3, 255, 254, 253])
        encoded_binary = self.valid_connector.encode_image(binary_data)
        decoded_binary = base64.b64decode(encoded_binary)
        self.assertEqual(decoded_binary, binary_data)

    def test_abstract_method_implementation(self):
        """Test that abstract methods are properly implemented"""
        # Test get_next_action
        result = self.valid_connector.get_next_action(
            self.test_image_data, "move right", ["UP", "DOWN"]
        )
        self.assertEqual(result, "RIGHT")

        # Test chat_with_ai
        chat_result = self.valid_connector.chat_with_ai(
            "Hello AI", self.test_image_data, {"goal": "test"}
        )
        self.assertIn("Hello AI", chat_result)

    def test_default_chat_implementation(self):
        """Test default chat implementation in base class"""
        # Create a minimal mock connector that doesn't override chat_with_ai
        class MinimalMockConnector(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                return "A"

        connector = MinimalMockConnector("test-key")
        result = connector.chat_with_ai("test", b"image", {})

        # Should return the default response
        self.assertIn("AI assistant", result)

    def test_logging_functionality(self):
        """Test that logging is properly configured"""
        with patch('logging.Logger.info') as mock_info, \
             patch('logging.Logger.error') as mock_error, \
             patch('logging.Logger.warning') as mock_warning, \
             patch('logging.Logger.debug') as mock_debug:

            # Test info logging
            self.valid_connector.get_next_action(
                self.test_image_data, "test goal", ["A", "B"]
            )
            mock_info.assert_called()

            # Test error logging
            try:
                self.failing_connector.get_next_action(
                    self.test_image_data, "test goal", ["A", "B"]
                )
            except:
                pass
            mock_error.assert_called()

    def test_action_history_handling(self):
        """Test handling of action history"""
        long_history = ["A"] * 1000  # Very long history

        result = self.valid_connector.get_next_action(
            self.test_image_data, "test goal", long_history
        )

        # Should handle long history without issues
        self.assertIsInstance(result, str)
        self.assertIn(result, {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"})

    def test_goal_parsing_logic(self):
        """Test different goal scenarios"""
        test_cases = [
            ("move to the right", "RIGHT"),
            ("go left", "LEFT"),
            ("jump up", "UP"),
            ("crouch down", "DOWN"),
            ("start game", "START"),
            ("select option", "SELECT"),
            ("attack", "A"),
            ("unknown goal", "A")
        ]

        for goal, expected_action in test_cases:
            with self.subTest(goal=goal):
                result = self.valid_connector.get_next_action(
                    self.test_image_data, goal, ["A", "B"]
                )
                self.assertEqual(result, expected_action)

    def test_image_data_handling(self):
        """Test handling of different image data scenarios"""
        # Test with small image data
        small_image = b"small"
        result = self.valid_connector.get_next_action(
            small_image, "test goal", ["A"]
        )
        self.assertIsInstance(result, str)

        # Test with large image data
        large_image = b"x" * 1000000  # 1MB of data
        result = self.valid_connector.get_next_action(
            large_image, "test goal", ["A"]
        )
        self.assertIsInstance(result, str)

        # Test with empty image data
        empty_image = b""
        result = self.valid_connector.get_next_action(
            empty_image, "test goal", ["A"]
        )
        self.assertIsInstance(result, str)

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading

        results = []

        def make_request():
            result = self.valid_connector.get_next_action(
                self.test_image_data, "concurrent test", ["A"]
            )
            results.append(result)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests were handled
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, str)

    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests with large data
        for i in range(100):
            large_image = b"x" * 100000  # 100KB per request
            self.valid_connector.get_next_action(
                large_image, f"test goal {i}", ["A"] * 100
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for this test)
        self.assertLess(memory_increase, 50 * 1024 * 1024)

    def test_error_propagation(self):
        """Test that errors are properly propagated"""
        with self.assertRaises(Exception):
            self.failing_connector.get_next_action(
                self.test_image_data, "test goal", ["A"]
            )

        with self.assertRaises(Exception):
            self.failing_connector.chat_with_ai(
                "test message", self.test_image_data, {}
            )

    def test_request_tracking(self):
        """Test that requests are properly tracked"""
        # Make a request
        self.valid_connector.get_next_action(
            self.test_image_data, "track this", ["A", "B"]
        )

        # Verify tracking
        self.assertEqual(self.valid_connector.call_count, 1)
        self.assertIsNotNone(self.valid_connector.last_request)
        self.assertEqual(self.valid_connector.last_request['goal'], "track this")

        # Make another request
        self.valid_connector.chat_with_ai(
            "hello", self.test_image_data, {"goal": "chat"}
        )

        # Verify tracking updated
        self.assertEqual(self.valid_connector.call_count, 2)
        self.assertEqual(self.valid_connector.last_request['message'], "hello")


class TestAIAPIBaseEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-api-key-12345"
        self.connector = ComprehensiveMockAIAPIConnector(self.test_api_key)

    def test_unicode_handling(self):
        """Test handling of unicode characters in goals and messages"""
        unicode_goal = "Move to 日本 (Japan) and then to Москва (Russia)"
        unicode_message = "Hello, 世界! How are you today?"

        result = self.connector.get_next_action(
            b"image_data", unicode_goal, ["A", "B"]
        )
        self.assertIsInstance(result, str)

        chat_result = self.connector.chat_with_ai(
            unicode_message, b"image_data", {"goal": unicode_goal}
        )
        self.assertIsInstance(chat_result, str)

    def test_special_characters_in_history(self):
        """Test handling of special characters in action history"""
        special_history = ["A", "B", "START", "SELECT", "↑", "↓", "←", "→"]

        result = self.connector.get_next_action(
            b"image_data", "test goal", special_history
        )
        self.assertIsInstance(result, str)

    def test_extremely_long_strings(self):
        """Test handling of extremely long strings"""
        long_goal = "test " * 10000  # 50,000 characters
        long_message = "hello " * 10000

        result = self.connector.get_next_action(
            b"image_data", long_goal, ["A"]
        )
        self.assertIsInstance(result, str)

        chat_result = self.connector.chat_with_ai(
            long_message, b"image_data", {"goal": "test"}
        )
        self.assertIsInstance(chat_result, str)

    def test_none_values_in_context(self):
        """Test handling of None values in context"""
        context_with_none = {
            "goal": None,
            "history": None,
            "game_type": None
        }

        result = self.connector.chat_with_ai(
            "test message", b"image_data", context_with_none
        )
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)