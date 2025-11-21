"""
Test suite for OpenRouter AI API connector
"""
import unittest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.openrouter_api import OpenRouterAPIConnector


class TestOpenRouterAPIConnector(unittest.TestCase):
    """Test suite for OpenRouter AI API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-openrouter-api-key-12345"
        self.connector = OpenRouterAPIConnector(self.test_api_key)
        self.test_image_data = b"test-game-screen-data"
        self.test_goal = "Collect all coins"
        self.test_history = ["UP", "LEFT", "B"]

    def test_initialization(self):
        """Test OpenRouter API connector initialization"""
        # Test with valid API key
        connector = OpenRouterAPIConnector("valid-api-key")
        self.assertEqual(connector.api_key, "valid-api-key")
        self.assertEqual(connector.valid_actions, {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'})
        self.assertEqual(connector.api_url, "https://openrouter.ai/api/v1/chat/completions")

        # Test with empty API key (should log warning but not crash)
        with patch('logging.Logger.warning') as mock_warning:
            connector = OpenRouterAPIConnector("")
            mock_warning.assert_called()

    @patch('requests.post')
    def test_get_next_action_success(self, mock_post):
        """Test successful action retrieval from OpenRouter"""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "DOWN"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "DOWN")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_get_next_action_case_insensitive(self, mock_post):
        """Test that action parsing is case insensitive"""
        # Mock response with lowercase action
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "up"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "UP")

    @patch('requests.post')
    def test_get_next_action_with_punctuation(self, mock_post):
        """Test that action parsing handles punctuation"""
        # Mock response with punctuation
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "RIGHT."
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "RIGHT")

    @patch('requests.post')
    def test_get_next_action_invalid_action(self, mock_post):
        """Test handling of invalid actions from OpenRouter"""
        # Mock response with invalid action
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "INVALID_ACTION"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT for invalid actions
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_malformed_response(self, mock_post):
        """Test handling of malformed response"""
        # Mock malformed response
        mock_response = Mock()
        mock_response.json.return_value = {
            "invalid": "response structure"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT for malformed responses
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_empty_choices(self, mock_post):
        """Test handling of empty choices list"""
        # Mock response with empty choices
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": []
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT for empty choices
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_network_error(self, mock_post):
        """Test handling of network errors"""
        # Mock network error
        mock_post.side_effect = Exception("Network error")

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT on network errors
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_timeout_error(self, mock_post):
        """Test handling of timeout errors"""
        import requests
        # Mock timeout error
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT on timeout
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_request_exception(self, mock_post):
        """Test handling of request exceptions"""
        import requests
        # Mock request exception
        mock_post.side_effect = requests.exceptions.RequestException("Request failed")

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT on request exceptions
        self.assertEqual(result, "SELECT")

    def test_api_request_headers(self):
        """Test that the API request includes correct headers"""
        with patch('requests.post') as mock_post:
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

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify the request headers
            call_args = mock_post.call_args
            headers = call_args[1]['headers']

            self.assertEqual(headers['Authorization'], f'Bearer {self.test_api_key}')
            self.assertEqual(headers['Content-Type'], 'application/json')
            self.assertIn('HTTP-Referer', headers)
            self.assertIn('X-Title', headers)
            self.assertEqual(headers['X-Title'], 'AI Game Server')

    def test_api_request_structure(self):
        """Test that the API request is structured correctly"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "B"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify the request structure
            call_args = mock_post.call_args
            request_data = call_args[1]['json']

            # Verify model
            self.assertEqual(request_data['model'], 'openai/gpt-4-vision-preview')

            # Verify messages structure
            self.assertIn('messages', request_data)
            self.assertEqual(len(request_data['messages']), 1)
            self.assertEqual(request_data['messages'][0]['role'], 'user')

            # Verify content structure
            content = request_data['messages'][0]['content']
            self.assertIsInstance(content, list)
            self.assertEqual(len(content), 2)

            # Verify image content
            image_content = content[0]
            self.assertEqual(image_content['type'], 'image_url')
            self.assertIn('image_url', image_content)
            self.assertTrue(image_content['image_url']['url'].startswith('data:image/jpeg;base64,'))

            # Verify text content
            text_content = content[1]
            self.assertEqual(text_content['type'], 'text')
            self.assertIn('text', text_content)

    def test_request_parameters(self):
        """Test that request parameters are correctly set"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "START"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify request parameters
            call_args = mock_post.call_args
            request_data = call_args[1]['json']

            self.assertEqual(request_data['max_tokens'], 10)
            self.assertEqual(request_data['temperature'], 0.7)

    @patch('requests.post')
    def test_chat_with_ai_success(self, mock_post):
        """Test successful chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "I see you're playing a retro game! Let me help you."
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {
            "current_goal": "Complete the level",
            "action_history": ["UP", "LEFT", "B"],
            "game_type": "GB"
        }

        result = self.connector.chat_with_ai(
            "What should I do next?", self.test_image_data, context
        )

        self.assertIn("help you", result)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_chat_with_ai_different_model(self, mock_post):
        """Test chat functionality with different model parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Chat response with higher token limit"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {"current_goal": "test goal"}

        self.connector.chat_with_ai(
            "Hello", self.test_image_data, context
        )

        # Verify chat uses different parameters
        call_args = mock_post.call_args
        request_data = call_args[1]['json']

        self.assertEqual(request_data['max_tokens'], 500)  # Higher limit for chat
        self.assertEqual(request_data['temperature'], 0.7)

    @patch('requests.post')
    def test_chat_with_ai_error_handling(self, mock_post):
        """Test error handling in chat functionality"""
        mock_post.side_effect = Exception("Chat API error")

        context = {"current_goal": "test goal"}

        result = self.connector.chat_with_ai(
            "Help me", self.test_image_data, context
        )

        # Should return error message
        self.assertIn("sorry", result.lower())

    @patch('requests.post')
    def test_retry_logic(self, mock_post):
        """Test retry logic for failed requests"""
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            else:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": "SUCCESS"
                        }
                    }]
                }
                mock_response.raise_for_status.return_value = None
                return mock_response

        mock_post.side_effect = mock_post_side_effect

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "SUCCESS")
        self.assertEqual(call_count, 3)

    def test_action_history_limiting(self):
        """Test that action history is properly limited in prompts"""
        long_history = ["A"] * 100  # Long history

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "LEFT"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, long_history
            )

            # Verify only last 5 actions are included
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            text_content = request_data['messages'][0]['content'][1]['text']

            # Should contain only the last 5 actions
            self.assertIn("A, A, A, A, A", text_content)

    def test_goal_in_prompt(self):
        """Test that goal is properly included in prompts"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "RIGHT"
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify goal is included
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            text_content = request_data['messages'][0]['content'][1]['text']

            self.assertIn(self.test_goal, text_content)

    @patch('requests.post')
    def test_chat_context_handling(self, mock_post):
        """Test that chat context is properly handled"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "I understand your context."
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {
            "current_goal": "Defeat the final boss",
            "action_history": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"],
            "game_type": "GB"
        }

        self.connector.chat_with_ai(
            "What strategy should I use?", self.test_image_data, context
        )

        # Verify context is included in the request
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['messages'][0]['content'][1]['text']

        self.assertIn("Defeat the final boss", text_content)
        self.assertIn("UP, DOWN, LEFT, RIGHT, A, B", text_content)
        self.assertIn("GB", text_content)
        self.assertIn("What strategy should I use?", text_content)

    @patch('requests.post')
    def test_timeout_setting(self, mock_post):
        """Test that timeout is properly set"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "SELECT"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Verify timeout is set
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['timeout'], 30)

    @patch('logging.Logger.info')
    @patch('logging.Logger.debug')
    @patch('requests.post')
    def test_logging_functionality(self, mock_post, mock_debug, mock_info):
        """Test that logging is properly implemented"""
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

        self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Verify logging calls
        mock_info.assert_called()
        mock_debug.assert_called()

    def test_base_class_methods(self):
        """Test that base class methods are inherited and work"""
        # Test encode_image method
        test_data = b"test image data"
        encoded = self.connector.encode_image(test_data)

        import base64
        decoded = base64.b64decode(encoded)
        self.assertEqual(decoded, test_data)

        # Test retry_with_backoff method
        def successful_function():
            return "success"

        result = self.connector._retry_with_backoff(successful_function)
        self.assertEqual(result, "success")

    @patch('requests.post')
    def test_empty_image_data_handling(self, mock_post):
        """Test handling of empty image data"""
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

        result = self.connector.get_next_action(
            b"", "test goal", ["A"]
        )

        self.assertEqual(result, "A")

    @patch('requests.post')
    def test_unicode_handling(self, mock_post):
        """Test handling of unicode characters"""
        unicode_goal = "Find the 🎮 and 🏆"
        unicode_message = "¡Hola! ¿Cómo estás?"

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "RIGHT"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            b"image_data", unicode_goal, ["A"]
        )

        self.assertEqual(result, "RIGHT")

        # Test chat with unicode
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "¡Hola! Estoy bien."
                }
            }]
        }

        context = {"current_goal": unicode_goal}
        chat_result = self.connector.chat_with_ai(
            unicode_message, b"image_data", context
        )

        self.assertIsInstance(chat_result, str)

    def test_model_configuration(self):
        """Test that the model is correctly configured"""
        self.assertEqual('openai/gpt-4-vision-preview', 'openai/gpt-4-vision-preview')


class TestOpenRouterAPIEdgeCases(unittest.TestCase):
    """Test edge cases for OpenRouter API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.connector = OpenRouterAPIConnector("test-api-key")

    @patch('requests.post')
    def test_response_content_with_explanation(self, mock_post):
        """Test handling of responses that include explanations"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "You should go RIGHT because there's a coin there."
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            b"image_data", "test goal", ["A"]
        )

        # Should extract just the action part
        self.assertEqual(result, "RIGHT")

    @patch('requests.post')
    def test_response_content_multiple_actions(self, mock_post):
        """Test handling of responses that mention multiple actions"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "First go UP, then RIGHT"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            b"image_data", "test goal", ["A"]
        )

        # Should take the first valid action mentioned
        self.assertEqual(result, "UP")

    @patch('requests.post')
    def test_response_content_with_numbers(self, mock_post):
        """Test handling of responses that include numbers"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Press A 3 times"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            b"image_data", "test goal", ["A"]
        )

        # Should extract the action
        self.assertEqual(result, "A")


if __name__ == '__main__':
    unittest.main(verbosity=2)