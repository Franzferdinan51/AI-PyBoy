"""
Test suite for NVIDIA NIM AI API connector
"""
import unittest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict
import numpy as np
from PIL import Image
import io

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.nvidia_api import NVIDIAAPIConnector


class TestNVIDIAAPIConnector(unittest.TestCase):
    """Test suite for NVIDIA NIM AI API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-nvidia-api-key-12345"
        self.connector = NVIDIAAPIConnector(self.test_api_key)
        self.test_image_data = self._create_test_image()
        self.test_goal = "Navigate to the exit"
        self.test_history = ["UP", "RIGHT", "A"]

    def _create_test_image(self):
        """Create test image data for testing"""
        # Create a simple test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        return img_buffer.getvalue()

    def test_initialization(self):
        """Test NVIDIA API connector initialization"""
        # Test with default model
        connector = NVIDIAAPIConnector("valid-api-key")
        self.assertEqual(connector.api_key, "valid-api-key")
        self.assertEqual(connector.valid_actions, {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'})
        self.assertEqual(connector.model, "nvidia/llama3-llm-70b")
        self.assertEqual(connector.api_url, "https://integrate.api.nvidia.com/v1/chat/completions")

        # Test with custom model
        connector = NVIDIAAPIConnector("valid-api-key", model="custom-model")
        self.assertEqual(connector.model, "custom-model")

        # Test with empty API key (should log warning but not crash)
        with patch('logging.Logger.warning') as mock_warning:
            connector = NVIDIAAPIConnector("")
            mock_warning.assert_called()

    def test_image_analysis_functionality(self):
        """Test the image analysis functionality"""
        # Create test images with different brightness levels
        dark_image = np.full((50, 50, 3), 50, dtype=np.uint8)  # Dark image
        bright_image = np.full((50, 50, 3), 200, dtype=np.uint8)  # Bright image
        normal_image = np.full((50, 50, 3), 128, dtype=np.uint8)  # Normal image

        # Test dark image analysis
        dark_img_buffer = io.BytesIO()
        Image.fromarray(dark_image).save(dark_img_buffer, format='JPEG')
        dark_bytes = dark_img_buffer.getvalue()

        bright_img_buffer = io.BytesIO()
        Image.fromarray(bright_image).save(bright_img_buffer, format='JPEG')
        bright_bytes = bright_img_buffer.getvalue()

        normal_img_buffer = io.BytesIO()
        Image.fromarray(normal_image).save(normal_img_buffer, format='JPEG')
        normal_bytes = normal_img_buffer.getvalue()

        # Test brightness detection
        self._test_brightness_level(dark_bytes, "dark")
        self._test_brightness_level(bright_bytes, "bright")
        self._test_brightness_level(normal_bytes, "normal")

    def _test_brightness_level(self, image_bytes, expected_level):
        """Helper method to test brightness level detection"""
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
                image_bytes, "test goal", ["A"]
            )

            # Verify brightness level is included in prompt
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            text_content = request_data['messages'][0]['content']

            self.assertIn(expected_level, text_content)

    @patch('requests.post')
    def test_get_next_action_success(self, mock_post):
        """Test successful action retrieval from NVIDIA NIM"""
        # Mock successful response
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

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "LEFT")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_get_next_action_case_insensitive(self, mock_post):
        """Test that action parsing is case insensitive"""
        # Mock response with lowercase action
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "down"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.connector.get_next_action(
            self.test_image_data, self.test_goal, self.test_history
        )

        self.assertEqual(result, "DOWN")

    @patch('requests.post')
    def test_get_next_action_invalid_action(self, mock_post):
        """Test handling of invalid actions from NVIDIA NIM"""
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

    def test_api_request_structure(self):
        """Test that the API request is structured correctly"""
        with patch('requests.post') as mock_post:
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

            # Verify the request structure
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            request_data = call_args[1]['json']

            # Verify headers
            self.assertEqual(headers['Authorization'], f'Bearer {self.test_api_key}')
            self.assertEqual(headers['Content-Type'], 'application/json')

            # Verify request data
            self.assertEqual(request_data['model'], self.connector.model)
            self.assertIn('messages', request_data)
            self.assertEqual(len(request_data['messages']), 1)
            self.assertEqual(request_data['messages'][0]['role'], 'user')

            # Verify prompt includes image analysis
            prompt_content = request_data['messages'][0]['content']
            self.assertIn(self.test_goal, prompt_content)
            self.assertIn("screen appears", prompt_content)  # Image analysis result

    def test_request_parameters(self):
        """Test that request parameters are correctly set"""
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

            # Verify request parameters
            call_args = mock_post.call_args
            request_data = call_args[1]['json']

            self.assertEqual(request_data['max_tokens'], 10)
            self.assertEqual(request_data['temperature'], 0.7)
            self.assertEqual(call_args[1]['timeout'], 30)

    @patch('requests.post')
    def test_chat_with_ai_success(self, mock_post):
        """Test successful chat interaction"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "I can help you with your retro game strategy!"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {
            "current_goal": "Complete the level",
            "action_history": ["UP", "RIGHT", "A"],
            "game_type": "GB"
        }

        result = self.connector.chat_with_ai(
            "What should I do next?", self.test_image_data, context
        )

        self.assertIn("help you", result)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_chat_with_ai_text_only_mode(self, mock_post):
        """Test that chat uses text-only mode (no image analysis)"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Based on your game context..."
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        context = {"current_goal": "test goal"}

        self.connector.chat_with_ai(
            "Hello", self.test_image_data, context
        )

        # Verify chat doesn't include image analysis
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        prompt_content = request_data['messages'][0]['content']

        # Should not contain image analysis
        self.assertNotIn("screen appears", prompt_content)
        self.assertIn("Based on the game context", prompt_content)

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

    def test_custom_model_initialization(self):
        """Test initialization with custom model"""
        custom_model = "nvidia/llama3-llm-8b"
        connector = NVIDIAAPIConnector("test-key", model=custom_model)

        self.assertEqual(connector.model, custom_model)

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

            connector.get_next_action(
                self.test_image_data, "test goal", ["A"]
            )

            # Verify custom model is used
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            self.assertEqual(request_data['model'], custom_model)

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
                        "content": "SELECT"
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
            text_content = request_data['messages'][0]['content']

            # Should contain only the last 5 actions
            self.assertIn("A, A, A, A, A", text_content)

    @patch('requests.post')
    def test_chat_context_handling(self, mock_post):
        """Test that chat context is properly handled"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "I understand your game situation."
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
            "What strategy should I use?", self.test_image_data, context
        )

        # Verify context is included in the request
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['messages'][0]['content']

        self.assertIn("Defeat the boss", text_content)
        self.assertIn("UP, DOWN, LEFT, RIGHT, A, B", text_content)
        self.assertIn("GB", text_content)
        self.assertIn("What strategy should I use?", text_content)

    @patch('requests.post')
    def test_image_analysis_error_handling(self, mock_post):
        """Test handling of image analysis errors"""
        # Mock a response to test the image analysis part
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

        # Test with corrupted image data
        corrupted_image = b"corrupted image data that cannot be parsed"

        # This should not raise an exception, but should handle it gracefully
        result = self.connector.get_next_action(
            corrupted_image, "test goal", ["A"]
        )

        # Should still get a valid action
        self.assertIn(result, self.connector.valid_actions)

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

    @patch('logging.Logger.info')
    @patch('logging.Logger.debug')
    @patch('requests.post')
    def test_logging_functionality(self, mock_post, mock_debug, mock_info):
        """Test that logging is properly implemented"""
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

        # Verify logging calls
        mock_info.assert_called()
        mock_debug.assert_called()

    @patch('requests.post')
    def test_different_image_formats(self, mock_post):
        """Test handling of different image formats"""
        # Create test images in different formats
        formats = ['JPEG', 'PNG', 'BMP']

        for img_format in formats:
            with self.subTest(format=img_format):
                # Create test image
                img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_buffer = io.BytesIO()
                img.save(img_buffer, format=img_format)
                image_bytes = img_buffer.getvalue()

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

                result = self.connector.get_next_action(
                    image_bytes, "test goal", ["A"]
                )

                self.assertEqual(result, "UP")


class TestNVIDIAAPIEdgeCases(unittest.TestCase):
    """Test edge cases for NVIDIA API connector"""

    def setUp(self):
        """Set up test fixtures"""
        self.connector = NVIDIAAPIConnector("test-api-key")

    def _create_test_image(self, brightness=128):
        """Create test image with specific brightness"""
        img_array = np.full((50, 50, 3), brightness, dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        return img_buffer.getvalue()

    @patch('requests.post')
    def test_boundary_brightness_values(self, mock_post):
        """Test brightness detection at boundary values"""
        # Test very dark (should be "dark")
        very_dark_image = self._create_test_image(brightness=0)
        self._test_brightness_in_prompt(very_dark_image, "dark", mock_post)

        # Test very bright (should be "bright")
        very_bright_image = self._create_test_image(brightness=255)
        self._test_brightness_in_prompt(very_bright_image, "bright", mock_post)

    def _test_brightness_in_prompt(self, image_bytes, expected_brightness, mock_post):
        """Helper to test brightness in prompt"""
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
            image_bytes, "test goal", ["A"]
        )

        # Verify brightness level is included
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['messages'][0]['content']

        self.assertIn(expected_brightness, text_content)

    @patch('requests.post')
    def test_empty_image_data(self, mock_post):
        """Test handling of empty image data"""
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

        result = self.connector.get_next_action(
            b"", "test goal", ["A"]
        )

        self.assertEqual(result, "B")

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

    @patch('requests.post')
    def test_extremely_long_context(self, mock_post):
        """Test handling of extremely long context"""
        long_goal = "test " * 1000
        long_history = ["ACTION"] * 200

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

        result = self.connector.get_next_action(
            b"image_data", long_goal, long_history
        )

        self.assertEqual(result, "SELECT")

        # Verify prompt is truncated appropriately
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        text_content = request_data['messages'][0]['content']

        # Should contain only recent history
        self.assertIn("ACTION, ACTION, ACTION, ACTION, ACTION", text_content)

    @patch('requests.post')
    def test_model_specific_configuration(self, mock_post):
        """Test different model configurations"""
        # Test with different models
        models = [
            "nvidia/llama3-llm-70b",
            "nvidia/llama3-llm-8b",
            "custom/nvidia-model"
        ]

        for model in models:
            with self.subTest(model=model):
                connector = NVIDIAAPIConnector("test-key", model=model)

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

                connector.get_next_action(
                    b"image_data", "test goal", ["A"]
                )

                # Verify correct model is used
                call_args = mock_post.call_args
                request_data = call_args[1]['json']
                self.assertEqual(request_data['model'], model)

    @patch('requests.post')
    def test_timeout_configuration(self, mock_post):
        """Test timeout configuration"""
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
            b"image_data", "test goal", ["A"]
        )

        # Verify timeout is set
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['timeout'], 30)


if __name__ == '__main__':
    unittest.main(verbosity=2)