"""
Integration tests for AI action execution pipeline with mock responses
"""
import unittest
import sys
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import threading
import queue
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


class MockEmulator:
    """Mock emulator for testing AI action pipeline"""

    def __init__(self):
        self.screen_data = self._create_test_screen()
        self.actions_executed = []
        self.frame_count = 0

    def _create_test_screen(self):
        """Create test screen data"""
        # Create a simple test image
        img_array = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        return img_array

    def get_screen(self):
        """Get current screen"""
        return self.screen_data

    def step(self, action, frames=1):
        """Execute action"""
        self.actions_executed.append((action, frames))
        self.frame_count += frames
        return True

    def reset(self):
        """Reset emulator state"""
        self.actions_executed = []
        self.frame_count = 0


class AIActionPipelineTest(unittest.TestCase):
    """Base class for AI action pipeline tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()
        self.test_image_data = self._get_emulator_image_bytes()
        self.test_goal = "Navigate to the exit"
        self.test_history = ["UP", "RIGHT", "A"]

    def _get_emulator_image_bytes(self):
        """Get emulator screen as bytes"""
        img_array = self.emulator.get_screen()
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        return img_buffer.getvalue()


class TestAIActionPipelineIntegration(AIActionPipelineTest):
    """Integration tests for AI action execution pipeline"""

    def test_action_pipeline_single_provider(self):
        """Test complete action pipeline with a single provider"""
        # Create mock provider
        class MockProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.responses = ["RIGHT", "DOWN", "LEFT", "A", "B"]
                self.call_count = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                action = self.responses[self.call_count % len(self.responses)]
                self.call_count += 1
                return action

        provider = MockProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Execute pipeline for multiple steps
        results = []
        for i in range(5):
            result = pipeline.execute_action_step(self.test_goal)
            results.append(result)

        # Verify all actions were executed
        self.assertEqual(len(results), 5)
        self.assertEqual(self.emulator.actions_executed, [
            ("RIGHT", 1), ("DOWN", 1), ("LEFT", 1), ("A", 1), ("B", 1)
        ])

        # Verify provider was called correct number of times
        self.assertEqual(provider.call_count, 5)

    def test_action_pipeline_with_retry(self):
        """Test action pipeline with retry mechanism"""
        class FailingThenSuccessProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.fail_count = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                if self.fail_count < 2:
                    self.fail_count += 1
                    raise Exception("Temporary failure")
                return "SUCCESS"

        provider = FailingThenSuccessProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Should succeed after retries
        result = pipeline.execute_action_step(self.test_goal)
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(provider.fail_count, 2)

    def test_action_pipeline_fallback_handling(self):
        """Test fallback handling when provider fails completely"""
        class AlwaysFailingProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                raise Exception("Always fails")

        provider = AlwaysFailingProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Should use fallback action
        result = pipeline.execute_action_step(self.test_goal)
        self.assertIn(result, {"UP", "RIGHT", "DOWN", "LEFT", "A", "B", "START", "SELECT"})

    def test_action_pipeline_history_tracking(self):
        """Test that action history is properly tracked"""
        class HistoryTrackingProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                # Return action based on history
                if not history or history[-1] != "RIGHT":
                    return "RIGHT"
                else:
                    return "LEFT"

        provider = HistoryTrackingProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Execute multiple actions
        result1 = pipeline.execute_action_step(self.test_goal)
        result2 = pipeline.execute_action_step(self.test_goal)
        result3 = pipeline.execute_action_step(self.test_goal)

        # Verify history influenced decisions
        self.assertEqual(result1, "RIGHT")
        self.assertEqual(result2, "LEFT")
        self.assertEqual(result3, "RIGHT")

    def test_action_pipeline_screen_capture_integration(self):
        """Test that screen capture is properly integrated"""
        class ScreenAwareProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.last_screen_data = None

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                self.last_screen_data = image_data
                return "A"

        provider = ScreenAwareProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        result = pipeline.execute_action_step(self.test_goal)

        # Verify screen data was captured and passed
        self.assertIsNotNone(provider.last_screen_data)
        self.assertEqual(result, "A")

    def test_action_pipeline_context_awareness(self):
        """Test that pipeline maintains context across actions"""
        class ContextAwareProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                # Check if goal and history are properly passed
                if goal and isinstance(history, list):
                    return "CONTEXT_OK"
                return "CONTEXT_MISSING"

        provider = ContextAwareProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        result = pipeline.execute_action_step(self.test_goal)

        self.assertEqual(result, "CONTEXT_OK")

    def test_action_pipeline_error_handling(self):
        """Test comprehensive error handling"""
        class ErrorProneProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.error_types = ["network", "timeout", "validation", "success"]
                self.error_index = 0

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                error_type = self.error_types[self.error_index % len(self.error_types)]
                self.error_index += 1

                if error_type == "network":
                    raise Exception("Network error")
                elif error_type == "timeout":
                    import time
                    time.sleep(2)  # Simulate timeout
                    return "TIMEOUT"
                elif error_type == "validation":
                    return "INVALID_ACTION"
                else:
                    return "SUCCESS"

        provider = ErrorProneProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        results = []
        for i in range(4):
            result = pipeline.execute_action_step(self.test_goal)
            results.append(result)

        # Verify all errors were handled
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertIn(result, {"UP", "RIGHT", "DOWN", "LEFT", "A", "B", "START", "SELECT", "TIMEOUT"})


class AIActionPipeline:
    """AI Action Pipeline for integration testing"""

    def __init__(self, provider: AIAPIConnector, emulator: MockEmulator):
        self.provider = provider
        self.emulator = emulator
        self.action_history = []
        self.execution_results = []

    def execute_action_step(self, goal: str) -> Dict[str, Any]:
        """Execute a single AI action step"""
        result = {
            'goal': goal,
            'success': False,
            'action': None,
            'error': None,
            'timestamp': time.time()
        }

        try:
            # Get screen from emulator
            screen_array = self.emulator.get_screen()
            if screen_array is None:
                result['error'] = "Failed to capture screen"
                return result

            # Convert screen to bytes
            img_buffer = io.BytesIO()
            Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()

            # Get action from AI provider
            action = self.provider.get_next_action(img_bytes, goal, self.action_history)

            # Validate action
            valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
            if action not in valid_actions:
                result['error'] = f"Invalid action: {action}"
                result['action'] = "SELECT"  # Fallback
            else:
                result['action'] = action

            # Execute action in emulator
            success = self.emulator.step(action, 1)
            if success:
                self.action_history.append(action)
                result['success'] = True
            else:
                result['error'] = "Failed to execute action in emulator"

        except Exception as e:
            result['error'] = str(e)
            # Use fallback action
            result['action'] = "UP"

        self.execution_results.append(result)
        return result

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution results"""
        total_actions = len(self.execution_results)
        successful_actions = sum(1 for r in self.execution_results if r['success'])
        failed_actions = total_actions - successful_actions

        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'failed_actions': failed_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'action_history': self.action_history.copy(),
            'errors': [r['error'] for r in self.execution_results if r['error']]
        }


class TestMultiProviderPipeline(AIActionPipelineTest):
    """Test pipeline with multiple providers"""

    def test_provider_fallback_chain(self):
        """Test fallback chain between multiple providers"""
        # Create providers with different failure patterns
        class PrimaryProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                raise Exception("Primary provider failed")

        class SecondaryProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                return "SECONDARY_SUCCESS"

        class FallbackProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                return "FALLBACK_SUCCESS"

        pipeline = MultiProviderPipeline([
            PrimaryProvider("primary"),
            SecondaryProvider("secondary"),
            FallbackProvider("fallback")
        ], self.emulator)

        result = pipeline.execute_with_fallback(self.test_goal)

        # Should have used secondary provider
        self.assertEqual(result['action'], "SECONDARY_SUCCESS")

    def test_concurrent_provider_testing(self):
        """Test concurrent execution with multiple providers"""
        class ConcurrentProvider(AIAPIConnector):
            def __init__(self, name, delay=0):
                super().__init__(name)
                self.name = name
                self.delay = delay

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                time.sleep(self.delay)
                return f"{self.name}_ACTION"

        providers = [
            ConcurrentProvider("fast", 0.1),
            ConcurrentProvider("medium", 0.3),
            ConcurrentProvider("slow", 0.5)
        ]

        pipeline = MultiProviderPipeline(providers, self.emulator)
        result = pipeline.execute_concurrent(self.test_goal)

        # Should return fastest response
        self.assertIn("fast", result['action'])


class MultiProviderPipeline:
    """Multi-provider pipeline for testing fallback and concurrent execution"""

    def __init__(self, providers: List[AIAPIConnector], emulator: MockEmulator):
        self.providers = providers
        self.emulator = emulator
        self.action_history = []

    def execute_with_fallback(self, goal: str) -> Dict[str, Any]:
        """Execute with fallback chain"""
        for i, provider in enumerate(self.providers):
            try:
                screen_array = self.emulator.get_screen()
                img_buffer = io.BytesIO()
                Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85)
                img_bytes = img_buffer.getvalue()

                action = provider.get_next_action(img_bytes, goal, self.action_history)

                # Try to execute the action
                success = self.emulator.step(action, 1)
                if success:
                    self.action_history.append(action)
                    return {
                        'success': True,
                        'action': action,
                        'provider_used': provider.__class__.__name__,
                        'provider_index': i
                    }
            except Exception as e:
                print(f"Provider {i} failed: {e}")
                continue

        # All providers failed
        return {
            'success': False,
            'action': 'SELECT',
            'error': 'All providers failed'
        }

    def execute_concurrent(self, goal: str) -> Dict[str, Any]:
        """Execute providers concurrently and return first successful response"""
        result_queue = queue.Queue()

        def worker(provider, provider_index):
            try:
                screen_array = self.emulator.get_screen()
                img_buffer = io.BytesIO()
                Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85)
                img_bytes = img_buffer.getvalue()

                action = provider.get_next_action(img_bytes, goal, self.action_history)
                result_queue.put((provider_index, action, None))
            except Exception as e:
                result_queue.put((provider_index, None, e))

        # Start all provider threads
        threads = []
        for i, provider in enumerate(self.providers):
            thread = threading.Thread(target=worker, args=(provider, i))
            threads.append(thread)
            thread.start()

        # Wait for first result or timeout
        try:
            provider_index, action, error = result_queue.get(timeout=2.0)

            if error:
                return {
                    'success': False,
                    'action': 'SELECT',
                    'error': str(error)
                }

            # Execute the successful action
            success = self.emulator.step(action, 1)
            if success:
                self.action_history.append(action)
                return {
                    'success': True,
                    'action': action,
                    'provider_index': provider_index
                }

        except queue.Empty:
            return {
                'success': False,
                'action': 'SELECT',
                'error': 'Timeout waiting for provider responses'
            }

        # Clean up threads
        for thread in threads:
            thread.join(timeout=0.1)

        return {
            'success': False,
            'action': 'SELECT',
            'error': 'No successful response'
        }


class TestRealProviderSimulation(AIActionPipelineTest):
    """Test pipeline with simulated real provider responses"""

    def test_gemini_simulation(self):
        """Test with simulated Gemini responses"""
        with patch('requests.post') as mock_post:
            # Mock successful Gemini response
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "RIGHT"}]
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = GeminiAPIConnector("test-gemini-key")
            pipeline = AIActionPipeline(provider, self.emulator)

            result = pipeline.execute_action_step(self.test_goal)

            self.assertTrue(result['success'])
            self.assertEqual(result['action'], 'RIGHT')

    def test_openrouter_simulation(self):
        """Test with simulated OpenRouter responses"""
        with patch('requests.post') as mock_post:
            # Mock successful OpenRouter response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "UP"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = OpenRouterAPIConnector("test-openrouter-key")
            pipeline = AIActionPipeline(provider, self.emulator)

            result = pipeline.execute_action_step(self.test_goal)

            self.assertTrue(result['success'])
            self.assertEqual(result['action'], 'UP')

    def test_nvidia_simulation(self):
        """Test with simulated NVIDIA responses"""
        with patch('requests.post') as mock_post:
            # Mock successful NVIDIA response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "A"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = NVIDIAAPIConnector("test-nvidia-key")
            pipeline = AIActionPipeline(provider, self.emulator)

            result = pipeline.execute_action_step(self.test_goal)

            self.assertTrue(result['success'])
            self.assertEqual(result['action'], 'A')

    @patch('openai.OpenAI')
    def test_openai_simulation(self, mock_openai):
        """Test with simulated OpenAI responses"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "B"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAICompatibleConnector("test-openai-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        result = pipeline.execute_action_step(self.test_goal)

        self.assertTrue(result['success'])
        self.assertEqual(result['action'], 'B')

    def test_mixed_provider_simulation(self):
        """Test pipeline with mixed provider types"""
        providers = []

        # Add Gemini
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "GEMINI_ACTION"}]}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            providers.append(GeminiAPIConnector("gemini-key"))

        # Add OpenRouter
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "OPENROUTER_ACTION"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            providers.append(OpenRouterAPIConnector("openrouter-key"))

        pipeline = MultiProviderPipeline(providers, self.emulator)
        result = pipeline.execute_with_fallback("mixed test")

        self.assertTrue(result['success'])
        self.assertIn(result['action'], ["GEMINI_ACTION", "OPENROUTER_ACTION"])


class TestPipelinePerformance(AIActionPipelineTest):
    """Test pipeline performance characteristics"""

    def test_performance_benchmark(self):
        """Test pipeline performance under load"""
        class FastProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                time.sleep(0.01)  # 10ms response time
                return "FAST"

        provider = FastProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Execute multiple actions and measure time
        start_time = time.time()
        results = []

        for i in range(10):
            result = pipeline.execute_action_step(f"goal_{i}")
            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance
        self.assertEqual(len(results), 10)
        self.assertLess(total_time, 1.0)  # Should complete in under 1 second
        self.assertTrue(all(r['success'] for r in results))

    def test_memory_usage(self):
        """Test memory usage during pipeline execution"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        class MemoryIntensiveProvider(AIAPIConnector):
            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                # Create some memory usage
                _ = [x for x in range(10000)]
                return "MEM_TEST"

        provider = MemoryIntensiveProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        # Execute many actions
        for i in range(50):
            pipeline.execute_action_step(f"memory_test_{i}")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        self.assertLess(memory_increase, 50 * 1024 * 1024)  # Less than 50MB

    def test_concurrent_execution_safety(self):
        """Test that pipeline handles concurrent execution safely"""
        class ConcurrentSafeProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.call_count = 0
                self.lock = threading.Lock()

            def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
                with self.lock:
                    self.call_count += 1
                    return f"CONCURRENT_{self.call_count}"

        provider = ConcurrentSafeProvider("test-key")
        pipeline = AIActionPipeline(provider, self.emulator)

        def worker(pipeline, goal, results):
            result = pipeline.execute_action_step(goal)
            results.append(result)

        # Execute concurrent actions
        results = []
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(pipeline, f"concurrent_goal_{i}", results))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all actions were executed successfully
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r['success'] for r in results))


if __name__ == '__main__':
    unittest.main(verbosity=2)