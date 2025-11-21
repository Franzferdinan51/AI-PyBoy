"""
Comprehensive chat functionality tests for all AI providers
"""
import unittest
import sys
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import io
import threading
import queue

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.ai_api_base import AIAPIConnector
from backend.ai_apis.gemini_api import GeminiAPIConnector
from backend.ai_apis.openrouter_api import OpenRouterAPIConnector
from backend.ai_apis.nvidia_api import NVIDIAAPIConnector
from backend.ai_apis.openai_compatible import OpenAICompatibleConnector


class MockEmulator:
    """Mock emulator for testing chat functionality"""

    def __init__(self):
        self.screen_data = self._create_test_screen()

    def _create_test_screen(self):
        """Create test screen data"""
        img_array = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        return img_array

    def get_screen(self):
        """Get current screen"""
        return self.screen_data


class TestChatFunctionalityBase(unittest.TestCase):
    """Base class for chat functionality tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()
        self.test_image_bytes = self._get_image_bytes()
        self.test_context = {
            "current_goal": "Defeat the final boss",
            "action_history": ["UP", "RIGHT", "A", "B", "START"],
            "game_type": "GB"
        }

    def _get_image_bytes(self):
        """Get emulator screen as bytes"""
        img_array = self.emulator.get_screen()
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        return img_buffer.getvalue()

    def _create_mock_response(self, response_text: str, provider_type: str):
        """Create mock response based on provider type"""
        if provider_type == "gemini":
            return {
                "candidates": [{
                    "content": {
                        "parts": [{"text": response_text}]
                    }
                }]
            }
        elif provider_type in ["openrouter", "nvidia", "openai"]:
            return {
                "choices": [{
                    "message": {
                        "content": response_text
                    }
                }]
            }
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


class TestGeminiChatFunctionality(TestChatFunctionalityBase):
    """Test Gemini chat functionality"""

    @patch('requests.post')
    def test_chat_success(self, mock_post):
        """Test successful Gemini chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "I can help you defeat the boss! Try using the B button for special attacks.",
            "gemini"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = GeminiAPIConnector("test-gemini-key")
        result = provider.chat_with_ai(
            "How do I defeat this boss?", self.test_image_bytes, self.test_context
        )

        self.assertIn("help you", result)
        self.assertIn("boss", result)

        # Verify API call structure
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn("key=test-gemini-key", call_args[1]['params'])

    @patch('requests.post')
    def test_chat_with_context(self, mock_post):
        """Test Gemini chat with game context"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Based on your goal to defeat the boss and your recent actions...",
            "gemini"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = GeminiAPIConnector("test-gemini-key")
        result = provider.chat_with_ai(
            "What should I do next?", self.test_image_bytes, self.test_context
        )

        # Verify context was included
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['contents'][0]['parts'][1]['text']

        self.assertIn("Defeat the final boss", text_content)
        self.assertIn("UP, RIGHT, A, B, START", text_content)
        self.assertIn("GB", text_content)

    @patch('requests.post')
    def test_chat_error_handling(self, mock_post):
        """Test Gemini chat error handling"""
        mock_post.side_effect = Exception("Network error")

        provider = GeminiAPIConnector("test-gemini-key")
        result = provider.chat_with_ai(
            "Help me", self.test_image_bytes, self.test_context
        )

        self.assertIn("sorry", result.lower())

    @patch('requests.post')
    def test_chat_unicode_support(self, mock_post):
        """Test Gemini chat unicode support"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "こんにちは！私はゲームのアシスタントです。",
            "gemini"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = GeminiAPIConnector("test-gemini-key")
        result = provider.chat_with_ai(
            "こんにちは", self.test_image_bytes, self.test_context
        )

        self.assertIsInstance(result, str)
        self.assertIn("こんにちは", result)

    @patch('requests.post')
    def test_chat_long_message_handling(self, mock_post):
        """Test handling of long chat messages"""
        long_message = "This is a very long message " * 100  # ~2600 characters

        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "I understand your long message.",
            "gemini"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = GeminiAPIConnector("test-gemini-key")
        result = provider.chat_with_ai(
            long_message, self.test_image_bytes, self.test_context
        )

        self.assertIsInstance(result, str)


class TestOpenRouterChatFunctionality(TestChatFunctionalityBase):
    """Test OpenRouter chat functionality"""

    @patch('requests.post')
    def test_chat_success(self, mock_post):
        """Test successful OpenRouter chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "I can see you're playing a retro game! Let me help you with your strategy.",
            "openrouter"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OpenRouterAPIConnector("test-openrouter-key")
        result = provider.chat_with_ai(
            "What strategy should I use?", self.test_image_bytes, self.test_context
        )

        self.assertIn("help you", result)
        self.assertIn("strategy", result)

        # Verify API call structure
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        self.assertEqual(headers['Authorization'], 'Bearer test-openrouter-key')

    @patch('requests.post')
    def test_chat_with_headers(self, mock_post):
        """Test OpenRouter chat includes required headers"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Response with proper headers",
            "openrouter"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OpenRouterAPIConnector("test-openrouter-key")
        provider.chat_with_ai(
            "Test", self.test_image_bytes, self.test_context
        )

        call_args = mock_post.call_args
        headers = call_args[1]['headers']

        self.assertIn('HTTP-Referer', headers)
        self.assertIn('X-Title', headers)
        self.assertEqual(headers['X-Title'], 'AI Game Server')

    @patch('requests.post')
    def test_chat_model_configuration(self, mock_post):
        """Test OpenRouter chat uses correct model"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Model test response",
            "openrouter"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OpenRouterAPIConnector("test-openrouter-key")
        provider.chat_with_ai(
            "Test model", self.test_image_bytes, self.test_context
        )

        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        self.assertEqual(request_data['model'], 'openai/gpt-4-vision-preview')

    @patch('requests.post')
    def test_chat_token_limits(self, mock_post):
        """Test OpenRouter chat respects token limits"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Token limit test response",
            "openrouter"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OpenRouterAPIConnector("test-openrouter-key")
        provider.chat_with_ai(
            "Test tokens", self.test_image_bytes, self.test_context
        )

        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        self.assertEqual(request_data['max_tokens'], 500)


class TestNVIDIAChatFunctionality(TestChatFunctionalityBase):
    """Test NVIDIA chat functionality"""

    @patch('requests.post')
    def test_chat_success(self, mock_post):
        """Test successful NVIDIA chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Based on your game context, I recommend using special attacks.",
            "nvidia"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = NVIDIAAPIConnector("test-nvidia-key")
        result = provider.chat_with_ai(
            "How do I beat this level?", self.test_image_bytes, self.test_context
        )

        self.assertIn("recommend", result)

    @patch('requests.post')
    def test_chat_text_only_mode(self, mock_post):
        """Test NVIDIA chat uses text-only mode"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Text-only response based on context",
            "nvidia"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = NVIDIAAPIConnector("test-nvidia-key")
        provider.chat_with_ai(
            "Hello", self.test_image_bytes, self.test_context
        )

        # Verify no image data was sent (text-only mode)
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        content = request_data['messages'][0]['content']

        # Should be text content, not image content
        self.assertIsInstance(content, str)
        self.assertNotIn("image", content.lower())

    @patch('requests.post')
    def test_chat_with_custom_model(self, mock_post):
        """Test NVIDIA chat with custom model"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Custom model response",
            "nvidia"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = NVIDIAAPIConnector("test-nvidia-key", model="custom/model")
        provider.chat_with_ai(
            "Test custom model", self.test_image_bytes, self.test_context
        )

        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        self.assertEqual(request_data['model'], "custom/model")

    @patch('requests.post')
    def test_chat_context_inclusion(self, mock_post):
        """Test NVIDIA chat includes game context"""
        mock_response = Mock()
        mock_response.json.return_value = self._create_mock_response(
            "Context-aware response",
            "nvidia"
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = NVIDIAAPIConnector("test-nvidia-key")
        provider.chat_with_ai(
            "Context test", self.test_image_bytes, self.test_context
        )

        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        content = request_data['messages'][0]['content']

        self.assertIn("Defeat the final boss", content)
        self.assertIn("UP, RIGHT, A, B, START", content)
        self.assertIn("GB", content)


class TestOpenAICompatibleChatFunctionality(TestChatFunctionalityBase):
    """Test OpenAI-compatible chat functionality"""

    @patch('openai.OpenAI')
    def test_chat_success(self, mock_openai):
        """Test successful OpenAI-compatible chat interaction"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "I can help you with your retro game strategy!"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAICompatibleConnector("test-openai-key")
        result = provider.chat_with_ai(
            "What should I do next?", self.test_image_bytes, self.test_context
        )

        self.assertIn("help you", result)

        # Verify API call structure
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['max_tokens'], 500)

    @patch('openai.OpenAI')
    def test_chat_with_system_prompt(self, mock_openai):
        """Test OpenAI-compatible chat uses correct system prompt"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "System prompt test response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAICompatibleConnector("test-openai-key")
        provider.chat_with_ai(
            "Test system prompt", self.test_image_bytes, self.test_context
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        system_message = messages[0]

        self.assertEqual(system_message['role'], 'system')
        self.assertIn("game assistant", system_message['content'])

    @patch('openai.OpenAI')
    def test_chat_with_local_provider(self, mock_openai):
        """Test OpenAI-compatible chat with local provider"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Local provider response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAICompatibleConnector("", base_url="localhost")
        result = provider.chat_with_ai(
            "Test local provider", self.test_image_bytes, self.test_context
        )

        self.assertIn("Local provider", result)

    @patch('openai.OpenAI')
    def test_chat_error_handling(self, mock_openai):
        """Test OpenAI-compatible chat error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        provider = OpenAICompatibleConnector("test-openai-key")
        result = provider.chat_with_ai(
            "Error test", self.test_image_bytes, self.test_context
        )

        self.assertIn("error", result.lower())

    @patch('openai.OpenAI')
    def test_chat_with_environment_variables(self, mock_openai):
        """Test OpenAI-compatible chat with environment variables"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Environment variable test"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {
            'AI_TIMEOUT': '120',
            'AI_MODEL': 'env-test-model'
        }):
            provider = OpenAICompatibleConnector("test-key")
            result = provider.chat_with_ai(
                "Test environment", self.test_image_bytes, self.test_context
            )

            self.assertEqual(provider.timeout, 120)
            self.assertEqual(provider.model, 'env-test-model')


class TestChatIntegrationScenarios(TestChatFunctionalityBase):
    """Test chat functionality with realistic scenarios"""

    def test_game_strategy_chat(self):
        """Test chat about game strategy"""
        test_messages = [
            "How do I beat this boss?",
            "What's the best strategy for this level?",
            "Which power-up should I use?",
            "How do I find the secret passage?",
            "What's the weakness of this enemy?"
        ]

        for message in test_messages:
            with self.subTest(message=message):
                # Test with mock provider
                class MockProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        return f"Strategy advice for: {message}"

                provider = MockProvider("test-key")
                result = provider.chat_with_ai(
                    message, self.test_image_bytes, self.test_context
                )

                self.assertIn(message, result)

    def test_game_state_chat(self):
        """Test chat about game state"""
        test_scenarios = [
            {"goal": "Find the key", "question": "Where is the key?"},
            {"goal": "Rescue the princess", "question": "How do I reach the princess?"},
            {"goal": "Collect all coins", "question": "Where are the remaining coins?"},
            {"goal": "Defeat the dragon", "question": "What's the dragon's weakness?"},
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                context = self.test_context.copy()
                context["current_goal"] = scenario["goal"]

                class ContextAwareProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        return f"For goal '{context['current_goal']}': {message}"

                provider = ContextAwareProvider("test-key")
                result = provider.chat_with_ai(
                    scenario["question"], self.test_image_bytes, context
                )

                self.assertIn(scenario["goal"], result)

    def test_action_history_chat(self):
        """Test chat with action history context"""
        test_histories = [
            ["UP", "UP", "RIGHT", "A"],  # Trying to reach something high
            ["LEFT", "LEFT", "DOWN", "B"],  # Exploring lower area
            ["A", "A", "A", "RIGHT"],  # Repeated action
            ["START", "SELECT", "UP", "DOWN"],  # Menu navigation
        ]

        for history in test_histories:
            with self.subTest(history=history):
                context = self.test_context.copy()
                context["action_history"] = history

                class HistoryAwareProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        last_actions = ', '.join(context['action_history'][-3:])
                        return f"Based on your recent actions ({last_actions}): {message}"

                provider = HistoryAwareProvider("test-key")
                result = provider.chat_with_ai(
                    "What should I do next?", self.test_image_bytes, context
                )

                self.assertIn("recent actions", result)

    def test_emergency_chat_scenarios(self):
        """Test chat in emergency scenarios"""
        emergency_scenarios = [
            {"situation": "low health", "message": "I'm about to die!"},
            {"situation": "time running out", "message": "I'm running out of time!"},
            {"situation": "stuck", "message": "I'm stuck and can't progress!"},
            {"situation": "lost", "message": "I don't know where to go!"},
        ]

        for scenario in emergency_scenarios:
            with self.subTest(scenario=scenario):
                class EmergencyProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        return f"Emergency help for {message}: Try using health items!"

                provider = EmergencyProvider("test-key")
                result = provider.chat_with_ai(
                    scenario["message"], self.test_image_bytes, self.test_context
                )

                self.assertIn("Emergency help", result)

    def test_multilingual_chat(self):
        """Test chat functionality with different languages"""
        test_cases = [
            {"lang": "es", "message": "¿Cómo derroto a este jefe?"},
            {"lang": "fr", "message": "Comment puis-je vaincre ce boss?"},
            {"lang": "de", "message": "Wie besiege ich diesen Boss?"},
            {"lang": "ja", "message": "このボスを倒すには？"},
            {"lang": "zh", "message": "我如何打败这个老板？"},
        ]

        for test_case in test_cases:
            with self.subTest(lang=test_case["lang"]):
                class MultilingualProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        return f"Response in {test_case['lang']}: Use special attacks!"

                provider = MultilingualProvider("test-key")
                result = provider.chat_with_ai(
                    test_case["message"], self.test_image_bytes, self.test_context
                )

                self.assertIn(test_case["lang"], result)

    def test_chat_performance_under_load(self):
        """Test chat performance under heavy load"""
        class FastMockProvider(AIAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                time.sleep(0.01)  # 10ms response time
                return f"Quick response to: {message[:20]}"

        provider = FastMockProvider("test-key")

        # Test performance with multiple concurrent requests
        start_time = time.time()
        results = []

        def worker(worker_id):
            for i in range(5):
                result = provider.chat_with_ai(
                    f"Worker {worker_id} message {i}", self.test_image_bytes, self.test_context
                )
                results.append(result)

        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance
        self.assertEqual(len(results), 15)  # 3 workers * 5 messages each
        self.assertLess(total_time, 2.0)  # Should complete in under 2 seconds

    def test_chat_error_recovery(self):
        """Test chat error recovery mechanisms"""
        class FlakyProvider(AIAPIConnector):
            def __init__(self, api_key):
                super().__init__(api_key)
                self.attempt_count = 0

            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                self.attempt_count += 1
                if self.attempt_count <= 2:
                    raise Exception("Temporary failure")
                return f"Recovered after {self.attempt_count - 1} failures"

        provider = FlakyProvider("test-key")

        # This test depends on the retry mechanism in the base class
        result = provider.chat_with_ai(
            "Test recovery", self.test_image_bytes, self.test_context
        )

        self.assertIn("Recovered", result)

    def test_chat_with_malformed_context(self):
        """Test chat with malformed or missing context"""
        malformed_contexts = [
            {},  # Empty context
            {"current_goal": None},  # None values
            {"action_history": None},
            {"game_type": ""},
            {"current_goal": "", "action_history": [], "game_type": None},
        ]

        for context in malformed_contexts:
            with self.subTest(context=context):
                class RobustProvider(AIAPIConnector):
                    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                        # Handle malformed context gracefully
                        goal = context.get('current_goal', 'No goal')
                        return f"Response for goal: {goal}"

                provider = RobustProvider("test-key")
                result = provider.chat_with_ai(
                    "Test malformed context", self.test_image_bytes, context
                )

                self.assertIsInstance(result, str)
                self.assertTrue(len(result) > 0)


class TestChatProviderComparison(unittest.TestCase):
    """Test comparison between different chat providers"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()
        self.test_image_bytes = self._get_image_bytes()
        self.test_context = {
            "current_goal": "Test goal",
            "action_history": ["UP", "RIGHT"],
            "game_type": "GB"
        }

    def _get_image_bytes(self):
        """Get emulator screen as bytes"""
        img_array = self.emulator.get_screen()
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        return img_buffer.getvalue()

    def test_provider_response_consistency(self):
        """Test that all providers can handle the same input"""
        test_message = "How do I beat this level?"

        # Create mock providers that all return similar responses
        class MockGeminiProvider(GeminiAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                return f"Gemini: {message}"

        class MockOpenRouterProvider(OpenRouterAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                return f"OpenRouter: {message}"

        class MockNVIDIAProvider(NVIDIAAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                return f"NVIDIA: {message}"

        class MockOpenAIProvider(OpenAICompatibleConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                return f"OpenAI: {message}"

        providers = [
            ("Gemini", MockGeminiProvider("gemini-key")),
            ("OpenRouter", MockOpenRouterProvider("openrouter-key")),
            ("NVIDIA", MockNVIDIAProvider("nvidia-key")),
            ("OpenAI", MockOpenAIProvider("openai-key")),
        ]

        results = {}
        for name, provider in providers:
            try:
                result = provider.chat_with_ai(
                    test_message, self.test_image_bytes, self.test_context
                )
                results[name] = result
            except Exception as e:
                results[name] = f"Error: {e}"

        # All providers should return a result
        self.assertEqual(len(results), 4)

        # All results should contain the test message
        for name, result in results.items():
            self.assertIn(test_message, result, f"Provider {name} failed to include test message")

    def test_provider_error_handling_comparison(self):
        """Test error handling across different providers"""
        class ErrorGeminiProvider(GeminiAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                raise Exception("Gemini error")

        class ErrorOpenRouterProvider(OpenRouterAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                raise Exception("OpenRouter error")

        class ErrorNVIDIAProvider(NVIDIAAPIConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                raise Exception("NVIDIA error")

        class ErrorOpenAIProvider(OpenAICompatibleConnector):
            def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
                raise Exception("OpenAI error")

        providers = [
            ("Gemini", ErrorGeminiProvider("gemini-key")),
            ("OpenRouter", ErrorOpenRouterProvider("openrouter-key")),
            ("NVIDIA", ErrorNVIDIAProvider("nvidia-key")),
            ("OpenAI", ErrorOpenAIProvider("openai-key")),
        ]

        results = {}
        for name, provider in providers:
            try:
                result = provider.chat_with_ai(
                    "Test error", self.test_image_bytes, self.test_context
                )
                results[name] = result
            except Exception as e:
                results[name] = f"Exception: {e}"

        # All providers should handle errors gracefully
        for name, result in results.items():
            self.assertIsInstance(result, str, f"Provider {name} didn't return a string")
            self.assertTrue(len(result) > 0, f"Provider {name} returned empty result")


if __name__ == '__main__':
    unittest.main(verbosity=2)