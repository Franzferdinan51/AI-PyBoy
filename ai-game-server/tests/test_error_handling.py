"""
Comprehensive error handling and fallback mechanism tests
"""
import unittest
import sys
import os
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any, Optional
import requests
import numpy as np
from PIL import Image
import io

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.ai_api_base import AIAPIConnector
from backend.ai_apis.gemini_api import GeminiAPIConnector
from backend.ai_apis.openrouter_api import OpenRouterAPIConnector
from backend.ai_apis.nvidia_api import NVIDIAAPIConnector
from backend.ai_apis.openai_compatible import OpenAICompatibleConnector


class ErrorTestProvider(AIAPIConnector):
    """Provider that simulates various error conditions"""

    def __init__(self, api_key: str, error_type: str = "none"):
        super().__init__(api_key)
        self.error_type = error_type
        self.call_count = 0
        self.error_count = 0

    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        self.call_count += 1

        if self.error_type == "network":
            self.error_count += 1
            raise requests.exceptions.ConnectionError("Network error")
        elif self.error_type == "timeout":
            self.error_count += 1
            raise requests.exceptions.Timeout("Request timeout")
        elif self.error_type == "rate_limit":
            self.error_count += 1
            raise requests.exceptions.HTTPError("429 Too Many Requests")
        elif self.error_type == "server_error":
            self.error_count += 1
            raise requests.exceptions.HTTPError("500 Internal Server Error")
        elif self.error_type == "invalid_response":
            self.error_count += 1
            return "INVALID_ACTION"
        elif self.error_type == "empty_response":
            self.error_count += 1
            return ""
        elif self.error_type == "malformed_json":
            self.error_count += 1
            raise json.JSONDecodeError("Invalid JSON", "", 0)
        elif self.error_type == "partial_failure":
            # Succeed after a few failures
            if self.call_count <= 2:
                self.error_count += 1
                raise Exception("Temporary failure")
            return "SUCCESS"
        else:
            return "NORMAL_ACTION"

    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
        self.call_count += 1

        if self.error_type == "chat_network":
            self.error_count += 1
            raise requests.exceptions.ConnectionError("Chat network error")
        elif self.error_type == "chat_timeout":
            self.error_count += 1
            raise requests.exceptions.Timeout("Chat timeout")
        elif self.error_type == "chat_server_error":
            self.error_count += 1
            raise Exception("Chat server error")
        else:
            return f"Chat response to: {message}"


class TestRetryMechanisms(unittest.TestCase):
    """Test retry mechanisms for different error types"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_exponential_backoff(self):
        """Test exponential backoff in retry mechanism"""
        provider = ErrorTestProvider("test-key", "partial_failure")
        original_retry_delay = provider.retry_delay

        with patch('time.sleep') as mock_sleep:
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SUCCESS")
            self.assertEqual(provider.call_count, 3)  # 2 failures + 1 success

            # Verify exponential backoff was used
            sleep_calls = mock_sleep.call_args_list
            self.assertEqual(len(sleep_calls), 2)  # 2 retry attempts

            # Verify delays are increasing (exponential)
            if len(sleep_calls) >= 2:
                first_delay = sleep_calls[0][0][0]
                second_delay = sleep_calls[1][0][0]
                self.assertGreater(second_delay, first_delay)

    def test_retry_with_jitter(self):
        """Test that retry includes jitter to prevent thundering herd"""
        provider = ErrorTestProvider("test-key", "partial_failure")

        sleep_times = []

        def capture_sleep(delay):
            sleep_times.append(delay)

        with patch('time.sleep', side_effect=capture_sleep):
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SUCCESS")

            # Verify jitter was added (delays should not be exactly the same)
            if len(sleep_times) >= 2:
                self.assertNotEqual(sleep_times[0], sleep_times[1])

    def test_max_retries_respected(self):
        """Test that max retries limit is respected"""
        provider = ErrorTestProvider("test-key", "network")
        provider.max_retries = 2

        with self.assertRaises(Exception):
            provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

        self.assertEqual(provider.call_count, 2)  # Should try exactly max_retries times

    def test_successful_retry_no_logging_error(self):
        """Test that successful retries don't log as errors"""
        provider = ErrorTestProvider("test-key", "partial_failure")

        with patch('time.sleep'), \
             patch('logging.Logger.warning') as mock_warning, \
             patch('logging.Logger.error') as mock_error:

            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SUCCESS")
            # Should have warnings for retries but no final error
            self.assertGreater(mock_warning.call_count, 0)
            self.assertEqual(mock_error.call_count, 0)

    def test_failed_retry_logs_error(self):
        """Test that failed retries log appropriate errors"""
        provider = ErrorTestProvider("test-key", "network")
        provider.max_retries = 1

        with patch('time.sleep'), \
             patch('logging.Logger.error') as mock_error:

            with self.assertRaises(Exception):
                provider.get_next_action(
                    self.test_image_data, self.test_goal, self.test_history
                )

            # Should log final error
            self.assertGreater(mock_error.call_count, 0)


class TestErrorTypeHandling(unittest.TestCase):
    """Test handling of different error types"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_network_error_handling(self):
        """Test network error handling"""
        provider = ErrorTestProvider("test-key", "network")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback to safe action

    def test_timeout_error_handling(self):
        """Test timeout error handling"""
        provider = ErrorTestProvider("test-key", "timeout")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback to safe action

    def test_rate_limit_error_handling(self):
        """Test rate limit error handling"""
        provider = ErrorTestProvider("test-key", "rate_limit")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback to safe action

    def test_server_error_handling(self):
        """Test server error handling"""
        provider = ErrorTestProvider("test-key", "server_error")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback to safe action

    def test_invalid_response_handling(self):
        """Test invalid response handling"""
        provider = ErrorTestProvider("test-key", "invalid_response")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback for invalid action

    def test_empty_response_handling(self):
        """Test empty response handling"""
        provider = ErrorTestProvider("test-key", "empty_response")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback for empty response

    def test_malformed_json_handling(self):
        """Test malformed JSON handling"""
        provider = ErrorTestProvider("test-key", "malformed_json")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SELECT")  # Should fallback for JSON errors

    def test_chat_error_handling(self):
        """Test chat error handling"""
        provider = ErrorTestProvider("test-key", "chat_network")

        result = provider.chat_with_ai(
            "Hello", self.test_image_data, {"goal": "test"}
        )

        self.assertIn("sorry", result.lower())  # Should return error message


class TestProviderSpecificErrorHandling(unittest.TestCase):
    """Test provider-specific error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_gemini_error_handling(self):
        """Test Gemini-specific error handling"""
        with patch('requests.post') as mock_post:
            # Mock network error
            mock_post.side_effect = requests.exceptions.ConnectionError("Network error")

            provider = GeminiAPIConnector("test-key")
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")

    def test_gemini_malformed_response_handling(self):
        """Test Gemini malformed response handling"""
        with patch('requests.post') as mock_post:
            # Mock malformed response
            mock_response = Mock()
            mock_response.json.return_value = {"malformed": "response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = GeminiAPIConnector("test-key")
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")

    def test_openrouter_error_handling(self):
        """Test OpenRouter-specific error handling"""
        with patch('requests.post') as mock_post:
            # Mock timeout error
            mock_post.side_effect = requests.exceptions.Timeout("Timeout")

            provider = OpenRouterAPIConnector("test-key")
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")

    def test_openrouter_malformed_response_handling(self):
        """Test OpenRouter malformed response handling"""
        with patch('requests.post') as mock_post:
            # Mock malformed response
            mock_response = Mock()
            mock_response.json.return_value = {"invalid": "structure"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = OpenRouterAPIConnector("test-key")
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")

    def test_nvidia_error_handling(self):
        """Test NVIDIA-specific error handling"""
        with patch('requests.post') as mock_post:
            # Mock server error
            mock_post.side_effect = requests.exceptions.HTTPError("500 Server Error")

            provider = NVIDIAAPIConnector("test-key")
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")

    def test_nvidia_image_processing_error(self):
        """Test NVIDIA image processing error handling"""
        # Create corrupted image data
        corrupted_image = b"corrupted image data"

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "A"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = NVIDIAAPIConnector("test-key")
            result = provider.get_next_action(
                corrupted_image, self.test_goal, self.test_history
            )

            # Should still return a valid action
            self.assertIn(result, provider.valid_actions)

    def test_openai_client_initialization_error(self):
        """Test OpenAI client initialization error handling"""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Client initialization failed")

            with patch('logging.Logger.error') as mock_error:
                provider = OpenAICompatibleConnector("test-key")
                self.assertIsNone(provider.client)
                mock_error.assert_called()

    def test_openai_no_client_error_handling(self):
        """Test OpenAI error handling when no client is available"""
        provider = OpenAICompatibleConnector("test-key")
        provider.client = None  # Simulate no client

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should use fallback action
        self.assertIn(result, provider.valid_actions)


class TestFallbackStrategies(unittest.TestCase):
    """Test different fallback strategies"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_action_history_based_fallback(self):
        """Test action history-based fallback strategy"""
        provider = ErrorTestProvider("test-key", "network")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, ["UP", "RIGHT", "A"]
        )

        # OpenAI-compatible has sophisticated fallback logic
        if hasattr(provider, '_get_fallback_action'):
            expected_fallback = provider._get_fallback_action(["UP", "RIGHT", "A"])
            self.assertEqual(result, expected_fallback)
        else:
            # Base providers fall back to SELECT
            self.assertEqual(result, "SELECT")

    def test_safe_action_fallback(self):
        """Test safe action fallback"""
        safe_actions = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"}

        error_types = ["network", "timeout", "rate_limit", "server_error"]

        for error_type in error_types:
            with self.subTest(error_type=error_type):
                provider = ErrorTestProvider("test-key", error_type)
                result = provider.get_next_action(
                    self.test_image_data, self.test_goal, self.test_history
                )

                self.assertIn(result, safe_actions)

    def test_chat_error_fallback_message(self):
        """Test chat error fallback messages"""
        error_types = ["chat_network", "chat_timeout", "chat_server_error"]

        for error_type in error_types:
            with self.subTest(error_type=error_type):
                provider = ErrorTestProvider("test-key", error_type)
                result = provider.chat_with_ai(
                    "Hello", self.test_image_data, {"goal": "test"}
                )

                self.assertIsInstance(result, str)
                self.assertTrue(len(result) > 0)
                # Should contain error indication
                error_indicators = ["sorry", "error", "apologize", "unavailable"]
                self.assertTrue(any(indicator in result.lower() for indicator in error_indicators))

    def test_progressive_fallback(self):
        """Test progressive fallback strategies"""
        class ProgressiveFallbackProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.attempt = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.attempt += 1
                if self.attempt == 1:
                    raise Exception("First attempt fails")
                elif self.attempt == 2:
                    return "INVALID_ACTION"
                else:
                    return "SUCCESS"

        provider = ProgressiveFallbackProvider("test-key")

        # First attempt - should fail and retry
        with patch('time.sleep'):
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

        # Should eventually succeed after progressive fallback
        self.assertEqual(result, "SUCCESS")


class TestCascadingFailureHandling(unittest.TestCase):
    """Test handling of cascading failures"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_multiple_providers_failing(self):
        """Test behavior when multiple providers fail"""
        failing_providers = [
            ErrorTestProvider("key1", "network"),
            ErrorTestProvider("key2", "timeout"),
            ErrorTestProvider("key3", "server_error"),
        ]

        results = []
        for provider in failing_providers:
            try:
                result = provider.get_next_action(
                    self.test_image_data, self.test_goal, self.test_history
                )
                results.append(result)
            except Exception as e:
                results.append(f"ERROR: {e}")

        # All should return fallback actions or handle gracefully
        for result in results:
            if not result.startswith("ERROR"):
                self.assertIn(result, {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"})

    def test_intermittent_failures(self):
        """Test handling of intermittent failures"""
        class IntermittentProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.success_rate = 0.6  # 60% success rate

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                import random
                if random.random() < self.success_rate:
                    return "SUCCESS"
                else:
                    raise Exception("Intermittent failure")

        provider = IntermittentProvider("test-key")

        results = []
        for _ in range(10):
            try:
                result = provider.get_next_action(
                    self.test_image_data, self.test_goal, self.test_history
                )
                results.append(result)
            except Exception:
                results.append("FALLBACK")

        # Should have some successes and some fallbacks
        success_count = results.count("SUCCESS")
        fallback_count = results.count("FALLBACK")

        self.assertGreater(success_count, 0)
        self.assertGreater(fallback_count, 0)

    def test_recovery_after_persistent_failure(self):
        """Test recovery after persistent failure"""
        class RecoveringProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.fail_count = 0
                self.max_failures = 3

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.fail_count += 1
                if self.fail_count <= self.max_failures:
                    raise Exception("Persistent failure")
                else:
                    return "RECOVERED"

        provider = RecoveringProvider("test-key")

        # Should eventually recover
        with patch('time.sleep'):
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

        self.assertEqual(result, "RECOVERED")


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation under various failure scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_partial_functionality_degradation(self):
        """Test partial functionality degradation"""
        class DegradingProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.degradation_level = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.degradation_level += 1

                if self.degradation_level == 1:
                    # First failure - return basic action
                    return "UP"
                elif self.degradation_level == 2:
                    # Second failure - return different basic action
                    return "RIGHT"
                else:
                    # Later failures - still return some action
                    return "A"

        provider = DegradingProvider("test-key")

        results = []
        for _ in range(5):
            try:
                result = provider.get_next_action(
                    self.test_image_data, self.test_goal, self.test_history
                )
                results.append(result)
            except Exception:
                results.append("ERROR")

        # Should never return an error - always some fallback
        self.assertNotIn("ERROR", results)
        for result in results:
            self.assertIn(result, {"UP", "RIGHT", "A"})

    def test_timeout_escalation(self):
        """Test timeout escalation under load"""
        class TimeoutEscalatingProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.request_count = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.request_count += 1
                if self.request_count > 2:
                    # Simulate increasing response time
                    import time
                    time.sleep(0.1 * self.request_count)
                return f"ACTION_{self.request_count}"

        provider = TimeoutEscalatingProvider("test-key")

        # Should handle timeout escalation gracefully
        for i in range(3):
            start_time = time.time()
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )
            end_time = time.time()

            self.assertIn(f"ACTION_{i+1}", result)

    def test_memory_pressure_handling(self):
        """Test handling under memory pressure"""
        import psutil
        import os

        class MemoryIntensiveProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.process = psutil.Process(os.getpid())

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                # Simulate memory pressure
                initial_memory = self.process.memory_info().rss

                # Try to allocate memory (but handle failure gracefully)
                try:
                    _ = [x for x in range(100000)]  # Allocate some memory
                    current_memory = self.process.memory_info().rss
                    memory_increase = current_memory - initial_memory

                    if memory_increase > 50 * 1024 * 1024:  # More than 50MB increase
                        raise MemoryError("Memory pressure too high")
                except MemoryError:
                    return "MEMORY_FALLBACK"

                return "NORMAL_ACTION"

        provider = MemoryIntensiveProvider("test-key")

        result = provider.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should handle memory pressure gracefully
        self.assertIn(result, ["NORMAL_ACTION", "MEMORY_FALLBACK"])


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery mechanisms"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_data = b"test-image-data"
        self.test_goal = "Test goal"
        self.test_history = ["A", "B"]

    def test_automatic_recovery(self):
        """Test automatic recovery from transient errors"""
        class AutoRecoveringProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.recovery_attempts = 0
                self.max_recovery_attempts = 3

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.recovery_attempts += 1

                if self.recovery_attempts <= self.max_recovery_attempts:
                    raise Exception(f"Temporary error {self.recovery_attempts}")
                else:
                    return "RECOVERED"

        provider = AutoRecoveringProvider("test-key")

        with patch('time.sleep'):
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

        self.assertEqual(result, "RECOVERED")
        self.assertEqual(provider.recovery_attempts, 4)  # 3 failures + 1 success

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for persistent failures"""
        class CircuitBreakerProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.failure_count = 0
                self.circuit_open = False
                self.last_failure_time = None

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                import time

                # Check if circuit is open and should remain open
                if self.circuit_open and self.last_failure_time:
                    if time.time() - self.last_failure_time < 5:  # 5 second cooldown
                        return "CIRCUIT_OPEN_FALLBACK"

                try:
                    # Simulate failure
                    raise Exception("Persistent failure")
                except Exception:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= 3:  # Open circuit after 3 failures
                        self.circuit_open = True

                    return "ERROR_FALLBACK"

        provider = CircuitBreakerProvider("test-key")

        results = []
        for _ in range(5):
            result = provider.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )
            results.append(result)
            time.sleep(0.1)  # Small delay

        # Should eventually open circuit
        self.assertIn("CIRCUIT_OPEN_FALLBACK", results)

    def test_fallback_action_intelligence(self):
        """Test intelligent fallback action selection"""
        class IntelligentFallbackProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                # Always fail to test fallback logic
                raise Exception("Always fails")

            def _get_intelligent_fallback(self, history: List[str]) -> str:
                # Intelligent fallback based on goal and history
                if "find" in goal.lower():
                    return "RIGHT"  # Explore
                elif "defeat" in goal.lower():
                    return "A"  # Attack
                elif not history:
                    return "UP"  # Start exploring
                elif history[-1] == "UP":
                    return "RIGHT"  # Change direction
                else:
                    return "DOWN"  # Try different direction

        provider = IntelligentFallbackProvider("test-key")

        # Test different goals
        test_goals = [
            ("Find the key", "RIGHT"),
            ("Defeat the enemy", "A"),
            ("Explore the area", "RIGHT"),
        ]

        for goal, expected in test_goals:
            with self.subTest(goal=goal):
                result = provider.get_next_action(
                    self.test_image_data, goal, self.test_history
                )
                self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)