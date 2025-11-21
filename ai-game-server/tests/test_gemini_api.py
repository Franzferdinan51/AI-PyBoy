"""
Test suite for Gemini AI API connector
"""
import unittest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import List, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.gemini_api import GeminiAPIConnector


class TestGeminiAPIConnector(unittest.TestCase):
    """Test suite for Gemini AI API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-gemini-api-key-12345"
        self.connector = GeminiAPIConnector(self.test_api_key)
        self.test_image_data = b"test-game-screen-data"
        self.test_goal = "Find the princess"
        self.test_history = ["UP", "RIGHT", "A"]

    def test_initialization(self):
        """Test Gemini API connector initialization"""
        # Test with valid API key
        connector = GeminiAPIConnector("valid-api-key")
        self.assertEqual(connector.api_key, "valid-api-key")
        self.assertEqual(connector.valid_actions, {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'})
        self.assertTrue(connector.api_url.startswith("https://generativelanguage.googleapis.com"))

        # Test with empty API key (should log warning but not crash)
        with patch('logging.Logger.warning') as mock_warning:
            connector = GeminiAPIConnector("")
            mock_warning.assert_called()

    @patch('requests.post')
    def test_get_next_action_success(self, mock_post):
        """Test successful action retrieval from Gemini"""
        # Mock successful response
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

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "RIGHT")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_get_next_action_case_insensitive(self, mock_post):
        """Test that action parsing is case insensitive"""
        # Mock response with lowercase action
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "left"}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "LEFT")

    @patch('requests.post')
    def test_get_next_action_invalid_action(self, mock_post):
        """Test handling of invalid actions from Gemini"""
        # Mock response with invalid action
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "INVALID_ACTION"}]
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
    def test_get_next_action_empty_candidates(self, mock_post):
        """Test handling of empty candidates list"""
        # Mock response with empty candidates
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": []
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT for empty candidates
        self.assertEqual(result, "SELECT")

    @patch('requests.post')
    def test_get_next_action_empty_parts(self, mock_post):
        """Test handling of empty parts list"""
        # Mock response with empty parts
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": []
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        # Should default to SELECT for empty parts
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

    @patch('requests.post')
    def test_get_next_action_retry_logic(self, mock_post):
        """Test retry logic for failed requests"""
        # Mock failure then success
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            else:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": "UP"}]
                        }
                    }]
                }
                mock_response.raise_for_status.return_value = None
                return mock_response

        mock_post.side_effect = mock_post_side_effect

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "UP")
        self.assertEqual(call_count, 3)  # 2 failures + 1 success

    def test_api_request_structure(self):
        """Test that the API request is structured correctly"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "A"}]
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
            self.assertIn('key=test-gemini-api-key-12345', call_args[1]['params'])
            self.assertEqual(call_args[1]['headers']['Content-Type'], 'application/json')

            # Verify the request body structure
            request_data = call_args[1]['json']
            self.assertIn('contents', request_data)
            self.assertEqual(len(request_data['contents']), 1)
            self.assertEqual(len(request_data['contents'][0]['parts']), 2)

            # Verify image data is included
            image_part = request_data['contents'][0]['parts'][0]
            self.assertIn('inline_data', image_part)
            self.assertEqual(image_part['inline_data']['mime_type'], 'image/jpeg')

            # Verify text prompt is included
            text_part = request_data['contents'][0]['parts'][1]
            self.assertIn('text', text_part)
            self.assertIn(self.test_goal, text_part['text'])

    @patch('requests.post')
    def test_chat_with_ai_success(self, mock_post):
        """Test successful chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello! I can help you with your game."}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {
            "current_goal": "Save the princess",
            "action_history": ["UP", "RIGHT", "A"],
            "game_type": "GB"
        }

        result = self.connector.chat_with_ai(
            "How do I beat this level?", self.test_image_data, context
        )

        self.assertIn("Hello", result)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_chat_with_ai_error_handling(self, mock_post):
        """Test error handling in chat functionality"""
        mock_post.side_effect = Exception("Chat API error")

        context = {
            "current_goal": "Save the princess",
            "action_history": ["UP", "RIGHT", "A"],
            "game_type": "GB"
        }

        result = self.connector.chat_with_ai(
            "How do I beat this level?", self.test_image_data, context
        )

        # Should return error message
        self.assertIn("sorry", result.lower())

    def test_encode_image_functionality(self):
        """Test image encoding functionality"""
        # Test that the base class encode_image method works
        test_data = b"test image data"
        encoded = self.connector.encode_image(test_data)

        self.assertIsInstance(encoded, str)
        import base64
        decoded = base64.b64decode(encoded)
        self.assertEqual(decoded, test_data)

    def test_action_history_limiting(self):
        """Test that action history is properly limited in prompts"""
        long_history = ["A"] * 100  # Long history

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "UP"}]
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
            text_content = request_data['contents'][0]['parts'][1]['text']

            # Should contain only the last 5 actions
            self.assertIn("A, A, A, A, A", text_content)

    def test_goal_and_context_in_prompt(self):
        """Test that goal and context are properly included in prompts"""
        with patch('requests.post') as mock_post:
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

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify goal is included
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            text_content = request_data['contents'][0]['parts'][1]['text']

            self.assertIn(self.test_goal, text_content)
            self.assertIn("UP, RIGHT, A", text_content)

    @patch('requests.post')
    def test_chat_context_handling(self, mock_post):
        """Test that chat context is properly handled"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "I understand your question."}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {
            "current_goal": "Defeat the boss",
            "action_history": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"],
            "game_type": "GB"
        }

        self.connector.chat_with_ai(
            "What should I do next?", self.test_image_data, context
        )

        # Verify context is included in the request
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['contents'][0]['parts'][1]['text']

        self.assertIn("Defeat the boss", text_content)
        self.assertIn("UP, DOWN, LEFT, RIGHT, A, B", text_content)
        self.assertIn("GB", text_content)
        self.assertIn("What should I do next?", text_content)

    @patch('time.sleep')
    @patch('requests.post')
    def test_retry_with_exponential_backoff(self, mock_post, mock_sleep):
        """Test that retry logic uses exponential backoff"""
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            else:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": "SUCCESS"}]
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

        # Verify sleep was called with exponential backoff
        self.assertEqual(mock_sleep.call_count, 2)  # 2 retries

    def test_different_api_endpoints(self):
        """Test that different API endpoints are used for different operations"""
        self.assertIn("gemini-pro-vision", self.connector.api_url)
        self.assertIn("gemini-pro", self.connector.text_api_url)

        # Verify endpoints are different
        self.assertNotEqual(self.connector.api_url, self.connector.text_api_url)

    @patch('requests.post')
    def test_logging_functionality(self, mock_post):
        """Test that logging is properly implemented"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "UP"}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch('logging.Logger.info') as mock_info, \
             patch('logging.Logger.debug') as mock_debug, \
             patch('logging.Logger.warning') as mock_warning, \
             patch('logging.Logger.error') as mock_error:

            self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            # Verify logging calls
            mock_info.assert_called()
            mock_debug.assert_called()

    @patch('requests.post')
    def test_request_timeout_handling(self, mock_post):
        """Test that request timeouts are properly handled"""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        with patch('logging.Logger.error') as mock_error:
            result = self.connector.get_next_action(
                self.test_image_data, self.test_goal, self.test_history
            )

            self.assertEqual(result, "SELECT")
            mock_error.assert_called()


class TestGeminiAPIEdgeCases(unittest.TestCase):
    """Test edge cases for Gemini API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.connector = GeminiAPIConnector("test-api-key")

    def test_empty_image_data(self):
        """Test handling of empty image data"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "A"}]
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = self.connector.get_next_action(
                b"", "test goal", ["A"]
            )

            self.assertEqual(result, "A")

    def test_unicode_in_goal_and_message(self):
        """Test handling of unicode characters"""
        unicode_goal = "Find the 日本の princess"
        unicode_message = "こんにちは、元気ですか？"

        with patch('requests.post') as mock_post:
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

            result = self.connector.get_next_action(
                b"image_data", unicode_goal, ["A"]
            )

            self.assertEqual(result, "RIGHT")

            # Test chat with unicode
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "こんにちは！"}]
                    }
                }]
            }

            context = {"current_goal": unicode_goal}
            chat_result = self.connector.chat_with_ai(
                unicode_message, b"image_data", context
            )

            self.assertIsInstance(chat_result, str)

    def test_extremely_long_history(self):
        """Test handling of extremely long action history"""
        long_history = ["A"] * 1000

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "B"}]
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = self.connector.get_next_action(
                b"image_data", "test goal", long_history
            )

            self.assertEqual(result, "B")


if __name__ == '__main__':
    unittest.main(verbosity=2)