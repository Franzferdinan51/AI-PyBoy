"""
Test suite for OpenAI-compatible API connector
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.openai_compatible import OpenAICompatibleConnector


class TestOpenAICompatibleConnector(unittest.TestCase):
    """Test suite for OpenAI-compatible API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-openai-api-key-12345"
        self.test_base_url = "https://api.openai.com/v1"
        self.test_model = "gpt-4o"
        self.connector = OpenAICompatibleConnector(
            api_key=self.test_api_key,
            base_url=self.test_base_url,
            model=self.test_model
        )
        self.test_image_data = b"test-game-screen-data"
        self.test_goal = "Defeat the boss"
        self.test_history = ["UP", "RIGHT", "B"]

    def test_initialization_default_values(self):
        """Test initialization with default values"""
        connector = OpenAICompatibleConnector("test-key")

        self.assertEqual(connector.api_key, "test-key")
        self.assertEqual(connector.base_url, "https://api.openai.com/v1")
        self.assertEqual(connector.model, "gpt-4o")
        self.assertEqual(connector.valid_actions, {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'})

    def test_initialization_custom_values(self):
        """Test initialization with custom values"""
        connector = OpenAICompatibleConnector(
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
            model="custom-model"
        )

        self.assertEqual(connector.api_key, "custom-key")
        self.assertEqual(connector.base_url, "https://custom.api.com/v1")
        self.assertEqual(connector.model, "custom-model")

    def test_base_url_validation(self):
        """Test base URL validation and normalization"""
        # Test with None base_url
        connector = OpenAICompatibleConnector("test-key", base_url=None)
        self.assertEqual(connector.base_url, "https://api.openai.com/v1")

        # Test with empty base_url
        connector = OpenAICompatibleConnector("test-key", base_url="")
        self.assertEqual(connector.base_url, "https://api.openai.com/v1")

        # Test with trailing slash removal
        connector = OpenAICompatibleConnector("test-key", base_url="https://api.test.com/v1/")
        self.assertEqual(connector.base_url, "https://api.test.com/v1")

        # Test LM Studio localhost conversion
        connector = OpenAICompatibleConnector("test-key", base_url="localhost")
        self.assertEqual(connector.base_url, "http://localhost:1234/v1")

        # Test LM Studio IP conversion
        connector = OpenAICompatibleConnector("test-key", base_url="127.0.0.1")
        self.assertEqual(connector.base_url, "http://127.0.0.1:1234/v1")

        # Test LM Studio named conversion
        connector = OpenAICompatibleConnector("test-key", base_url="lm-studio")
        self.assertEqual(connector.base_url, "http://lm-studio:1234/v1")

        # Test HTTP prefix addition
        connector = OpenAICompatibleConnector("test-key", base_url="custom-api.com")
        self.assertEqual(connector.base_url, "http://custom-api.com")

    def test_model_selection(self):
        """Test model selection based on provider"""
        # Test localhost model selection
        connector = OpenAICompatibleConnector("test-key", base_url="localhost")
        self.assertEqual(connector.model, "gpt-4-vision-preview")

        # Test LM Studio model selection
        connector = OpenAICompatibleConnector("test-key", base_url="lm-studio")
        self.assertEqual(connector.model, "local-model")

        # Test OpenAI model selection
        connector = OpenAICompatibleConnector("test-key", base_url="https://api.openai.com/v1")
        self.assertEqual(connector.model, "gpt-4o")

    @patch('openai.OpenAI')
    def test_client_initialization_success(self, mock_openai):
        """Test successful client initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        connector = OpenAICompatibleConnector("test-key")
        self.assertEqual(connector.client, mock_client)

        # Verify OpenAI was called with correct parameters
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            timeout=60
        )

    @patch('openai.OpenAI')
    def test_client_initialization_failure(self, mock_openai):
        """Test client initialization failure"""
        mock_openai.side_effect = Exception("Initialization failed")

        with patch('logging.Logger.error') as mock_error:
            connector = OpenAICompatibleConnector("test-key")
            self.assertIsNone(connector.client)
            mock_error.assert_called()

    @patch('openai.OpenAI')
    def test_connection_test_success(self, mock_openai):
        """Test successful connection test"""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.data = [Mock(), Mock()]  # Simulate 2 models
        mock_client.models.list.return_value = mock_models
        mock_openai.return_value = mock_client

        with patch('logging.Logger.info') as mock_info:
            connector = OpenAICompatibleConnector("test-key")
            mock_info.assert_called()
            self.assertIn("Available models: 2", str(mock_info.call_args))

    @patch('openai.OpenAI')
    def test_connection_test_failure(self, mock_openai):
        """Test connection test failure"""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Connection failed")
        mock_openai.return_value = mock_client

        with patch('logging.Logger.warning') as mock_warning:
            connector = OpenAICompatibleConnector("test-key")
            mock_warning.assert_called()

    @patch('openai.OpenAI')
    def test_get_next_action_no_client(self, mock_openai):
        """Test get_next_action when no client is available"""
        mock_openai.side_effect = Exception("No client")

        with patch('logging.Logger.error') as mock_error:
            connector = OpenAICompatibleConnector("test-key")
            result = connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "UP")  # Fallback action
            mock_error.assert_called()

    @patch('openai.OpenAI')
    def test_get_next_action_success(self, mock_openai):
        """Test successful action retrieval"""
        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock successful response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "RIGHT"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        connector = OpenAICompatibleConnector("test-key")
        result = connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "RIGHT")

        # Verify the API call was made correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], connector.model)
        self.assertEqual(call_args[1]['max_tokens'], 10)
        self.assertEqual(call_args[1]['temperature'], 0.7)

    @patch('openai.OpenAI')
    def test_get_next_action_with_punctuation(self, mock_openai):
        """Test action parsing with punctuation"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "LEFT."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        connector = OpenAICompatibleConnector("test-key")
        result = connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "LEFT")

    @patch('openai.OpenAI')
    def test_get_next_action_invalid_action(self, mock_openai):
        """Test handling of invalid actions"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "INVALID_ACTION"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        with patch('logging.Logger.warning') as mock_warning:
            connector = OpenAICompatibleConnector("test-key")
            result = connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Should use fallback action
            self.assertIn(result, connector.valid_actions)
            mock_warning.assert_called()

    @patch('openai.OpenAI')
    def test_get_next_action_api_error(self, mock_openai):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        with patch('logging.Logger.error') as mock_error:
            connector = OpenAICompatibleConnector("test-key")
            result = connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Should use fallback action
            self.assertIn(result, connector.valid_actions)
            mock_error.assert_called()

    @patch('openai.OpenAI')
    def test_request_structure(self, mock_openai):
        """Test that the API request is structured correctly"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "A"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        connector = OpenAICompatibleConnector("test-key")
        connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Verify request structure
        call_args = mock_client.chat.completions.create.call_args
        kwargs = call_args[1]

        # Verify basic parameters
        self.assertEqual(kwargs['model'], connector.model)
        self.assertEqual(kwargs['max_tokens'], 10)
        self.assertEqual(kwargs['temperature'], 0.7)
        self.assertEqual(kwargs['timeout'], connector.timeout)

        # Verify messages structure
        messages = kwargs['messages']
        self.assertEqual(len(messages), 2)

        # Verify system message
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('expert AI playing', messages[0]['content'])

        # Verify user message
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIsInstance(messages[1]['content'], list)
        self.assertEqual(len(messages[1]['content']), 2)

        # Verify text content
        text_content = messages[1]['content'][0]
        self.assertEqual(text_content['type'], 'text')
        self.assertIn(connector.test_goal, text_content['text'])

        # Verify image content
        image_content = messages[1]['content'][1]
        self.assertEqual(image_content['type'], 'image_url')
        self.assertTrue(image_content['image_url']['url'].startswith('data:image/jpeg;base64,'))

    def test_fallback_action_logic(self):
        """Test fallback action selection logic"""
        connector = OpenAICompatibleConnector("test-key")

        # Test with empty history
        result = connector._get_fallback_action([])
        self.assertEqual(result, "UP")

        # Test with different last actions
        test_cases = [
            (["UP"], "RIGHT"),
            (["RIGHT"], "DOWN"),
            (["DOWN"], "LEFT"),
            (["LEFT"], "A"),
            (["A"], "UP"),  # Should cycle back
            (["START"], "UP"),  # Should default to UP for non-movement actions
        ]

        for history, expected in test_cases:
            with self.subTest(history=history):
                result = connector._get_fallback_action(history)
                self.assertEqual(result, expected)

    @patch('openai.OpenAI')
    def test_chat_with_ai_success(self, mock_openai):
        """Test successful chat interaction"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Hello! I can help you with your game."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        connector = OpenAICompatibleConnector("test-key")
        context = {"current_goal": "Save the princess", "game_type": "GB"}

        result = connector.chat_with_ai(
            "How do I beat this level?", self.test_image_data, context
        )

        self.assertIn("help you", result)

        # Verify chat request structure
        call_args = mock_client.chat.completions.create.call_args
        kwargs = call_args[1]

        self.assertEqual(kwargs['max_tokens'], 500)  # Higher limit for chat
        self.assertEqual(kwargs['temperature'], 0.7)

        # Verify system message
        messages = kwargs['messages']
        self.assertIn("helpful game assistant", messages[0]['content'])

    @patch('openai.OpenAI')
    def test_chat_with_ai_no_client(self, mock_openai):
        """Test chat functionality when no client is available"""
        mock_openai.side_effect = Exception("No client")

        connector = OpenAICompatibleConnector("test-key")
        result = connector.chat_with_ai(
            "Hello", self.test_image_data, {}
        )

        self.assertIn("not available", result)

    @patch('openai.OpenAI')
    def test_chat_with_ai_error_handling(self, mock_openai):
        """Test error handling in chat functionality"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Chat error")
        mock_openai.return_value = mock_client

        with patch('logging.Logger.error') as mock_error:
            connector = OpenAICompatibleConnector("test-key")
            result = connector.chat_with_ai(
                "Help me", self.test_image_data, {}
            )

            self.assertIn("error", result.lower())
            mock_error.assert_called()

    def test_prompt_creation(self):
        """Test prompt creation functionality"""
        connector = OpenAICompatibleConnector("test-key")

        # Test action prompt creation
        goal = "Find the treasure"
        history = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]

        prompt = connector._create_action_prompt(goal, history)

        self.assertIn(goal, prompt)
        self.assertIn("UP, DOWN, LEFT, RIGHT, A, B", prompt)
        self.assertIn("expert AI playing", prompt)
        self.assertIn("MUST be one of the following", prompt)

        # Test chat prompt creation
        context = {
            "current_goal": "Defeat the boss",
            "action_history": ["A", "B", "START"],
            "game_type": "GBA"
        }

        chat_prompt = connector._create_chat_prompt("What should I do?", context)

        self.assertIn("What should I do?", chat_prompt)
        self.assertIn("Defeat the boss", chat_prompt)
        self.assertIn("A, B, START", chat_prompt)
        self.assertIn("GBA", chat_prompt)

    def test_system_prompts(self):
        """Test system prompt generation"""
        connector = OpenAICompatibleConnector("test-key")

        system_prompt = connector._get_system_prompt()
        self.assertIn("expert AI playing", system_prompt)
        self.assertIn("only the action name", system_prompt)

    def test_response_parsing(self):
        """Test response parsing functionality"""
        connector = OpenAICompatibleConnector("test-key")

        # Test normal response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "RIGHT"
        mock_response.choices = [mock_choice]

        result = connector._parse_action_response(mock_response)
        self.assertEqual(result, "RIGHT")

        # Test response with punctuation
        mock_choice.message.content = "LEFT."
        result = connector._parse_action_response(mock_response)
        self.assertEqual(result, "LEFT")

        # Test response with comma
        mock_choice.message.content = "UP,"
        result = connector._parse_action_response(mock_response)
        self.assertEqual(result, "UP")

        # Test malformed response
        mock_response.choices = []
        result = connector._parse_action_response(mock_response)
        self.assertEqual(result, "SELECT")

    @patch('openai.OpenAI')
    def test_retry_logic(self, mock_openai):
        """Test retry logic for failed requests"""
        mock_client = Mock()
        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            else:
                mock_response = Mock()
                mock_choice = Mock()
                mock_choice.message.content = "SUCCESS"
                mock_response.choices = [mock_choice]
                return mock_response

        mock_client.chat.completions.create = mock_create
        mock_openai.return_value = mock_client

        connector = OpenAICompatibleConnector("test-key")
        result = connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SUCCESS")
        self.assertEqual(call_count, 3)

    def test_environment_variable_support(self):
        """Test environment variable support"""
        with patch.dict(os.environ, {
            'AI_TIMEOUT': '120',
            'AI_MAX_RETRIES': '5',
            'AI_MODEL': 'custom-env-model',
            'OPENAI_ENDPOINT': 'https://env.api.com/v1'
        }):
            connector = OpenAICompatibleConnector("test-key")

            self.assertEqual(connector.timeout, 120)
            self.assertEqual(connector.max_retries, 5)
            self.assertEqual(connector.model, 'custom-env-model')

            # Test with None base_url (should use environment)
            connector2 = OpenAICompatibleConnector("test-key", base_url=None)
            self.assertEqual(connector2.base_url, "https://env.api.com/v1")

    def test_local_provider_api_key_handling(self):
        """Test API key handling for local providers"""
        # Test localhost provider
        connector = OpenAICompatibleConnector("", base_url="localhost")
        self.assertIsNotNone(connector.client)

        # Test 127.0.0.1 provider
        connector = OpenAICompatibleConnector(None, base_url="127.0.0.1")
        self.assertIsNotNone(connector.client)


class TestOpenAICompatibleEdgeCases(unittest.TestCase):
    """Test edge cases for OpenAI-compatible connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.connector = OpenAICompatibleConnector("test-key")

    @patch('openai.OpenAI')
    def test_empty_image_data(self, mock_openai):
        """Test handling of empty image data"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "A"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = self.connector.get_next_action(
            b"", "test goal", ["A"]
        )

        self.assertEqual(result, "A")

    @patch('openai.OpenAI')
    def test_unicode_handling(self, mock_openai):
        """Test handling of unicode characters"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "RIGHT"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        unicode_goal = "Find the 日本の princess"
        unicode_message = "こんにちは、元気ですか？"

        result = self.connector.get_next_action(
            b"image_data", unicode_goal, ["A"]
        )

        self.assertEqual(result, "RIGHT")

        # Test chat with unicode
        mock_choice.message.content = "こんにちは！"
        context = {"current_goal": unicode_goal}
        chat_result = self.connector.chat_with_ai(
            unicode_message, b"image_data", context
        )

        self.assertIsInstance(chat_result, str)

    @patch('openai.OpenAI')
    def test_long_context_handling(self, mock_openai):
        """Test handling of long context"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "B"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        long_goal = "test " * 1000
        long_history = ["ACTION"] * 100

        result = self.connector.get_next_action(
            b"image_data", long_goal, long_history
        )

        self.assertEqual(result, "B")

    @patch('openai.OpenAI')
    def test_response_content_variations(self, mock_openai):
        """Test handling of various response content formats"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        test_cases = [
            ("Press RIGHT", "RIGHT"),
            ("You should go LEFT", "LEFT"),
            ("UP and then JUMP", "UP"),  # Should take first action
            ("The button to press is DOWN.", "DOWN"),
            ("A button", "A"),
            ("   START   ", "START"),  # With whitespace
            ("select\n", "SELECT"),  # With newline
        ]

        for response_content, expected_action in test_cases:
            with self.subTest(response_content=response_content):
                mock_response = Mock()
                mock_choice = Mock()
                mock_choice.message.content = response_content
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create.return_value = mock_response

                connector = OpenAICompatibleConnector("test-key")
                result = connector.get_next_action(
                    b"image_data", "test goal", ["A"]
                )

                self.assertEqual(result, expected_action)

    @patch('openai.OpenAI')
    def test_timeout_configuration(self, mock_openai):
        """Test timeout configuration"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "START"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {'AI_TIMEOUT': '120'}):
            connector = OpenAICompatibleConnector("test-key")
            connector.get_next_action(
                b"image_data", "test goal", ["A"]
            )

            # Verify timeout is passed correctly
            call_args = mock_client.chat.completions.create.call_args
            self.assertEqual(call_args[1]['timeout'], 120)


if __name__ == '__main__':
    unittest.main(verbosity=2)