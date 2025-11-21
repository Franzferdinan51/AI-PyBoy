"""
Test suite for environment variable configuration
"""
import unittest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.ai_apis.gemini_api import GeminiAPIConnector
from backend.ai_apis.openrouter_api import OpenRouterAPIConnector
from backend.ai_apis.nvidia_api import NVIDIAAPIConnector
from backend.ai_apis.openai_compatible import OpenAICompatibleConnector
from backend.server import initialize_ai_apis


class TestEnvironmentVariableConfiguration(unittest.TestCase):
    """Test environment variable configuration for AI providers"""

    def setUp(self):
        """Set up test fixtures"""
        # Store original environment variables
        self.original_env = os.environ.copy()

        # Clear relevant environment variables for clean testing
        env_vars_to_clear = [
            'GEMINI_API_KEY',
            'OPENROUTER_API_KEY',
            'NVIDIA_API_KEY',
            'NVIDIA_MODEL',
            'OPENAI_API_KEY',
            'OPENAI_ENDPOINT',
            'AI_TIMEOUT',
            'AI_MAX_RETRIES',
            'AI_MODEL',
            'AI_ENDPOINT'
        ]

        for var in env_vars_to_clear:
            os.environ.pop(var, None)

    def tearDown(self):
        """Restore original environment variables"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_gemini_api_key_configuration(self):
        """Test Gemini API key configuration"""
        # Test with valid API key
        os.environ['GEMINI_API_KEY'] = 'test-gemini-key-12345'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "UP"}]}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = GeminiAPIConnector(os.environ['GEMINI_API_KEY'])
            self.assertEqual(provider.api_key, 'test-gemini-key-12345')

    def test_gemini_api_key_missing(self):
        """Test Gemini API key missing configuration"""
        with patch('logging.Logger.warning') as mock_warning:
            provider = GeminiAPIConnector("")
            mock_warning.assert_called()

    def test_openrouter_api_key_configuration(self):
        """Test OpenRouter API key configuration"""
        os.environ['OPENROUTER_API_KEY'] = 'test-openrouter-key-12345'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "RIGHT"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = OpenRouterAPIConnector(os.environ['OPENROUTER_API_KEY'])
            self.assertEqual(provider.api_key, 'test-openrouter-key-12345')

    def test_nvidia_api_key_configuration(self):
        """Test NVIDIA API key configuration"""
        os.environ['NVIDIA_API_KEY'] = 'test-nvidia-key-12345'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "A"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = NVIDIAAPIConnector(os.environ['NVIDIA_API_KEY'])
            self.assertEqual(provider.api_key, 'test-nvidia-key-12345')

    def test_nvidia_model_configuration(self):
        """Test NVIDIA model configuration"""
        os.environ['NVIDIA_API_KEY'] = 'test-nvidia-key'
        os.environ['NVIDIA_MODEL'] = 'nvidia/llama3-llm-8b'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "B"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = NVIDIAAPIConnector(os.environ['NVIDIA_API_KEY'])
            self.assertEqual(provider.model, 'nvidia/llama3-llm-8b')

    def test_nvidia_model_default(self):
        """Test NVIDIA model default configuration"""
        os.environ['NVIDIA_API_KEY'] = 'test-nvidia-key'
        # Don't set NVIDIA_MODEL, should use default

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "START"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider = NVIDIAAPIConnector(os.environ['NVIDIA_API_KEY'])
            self.assertEqual(provider.model, 'nvidia/llama3-llm-70b')  # Default model

    def test_openai_api_key_configuration(self):
        """Test OpenAI API key configuration"""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key-12345'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(os.environ['OPENAI_API_KEY'])
            self.assertEqual(provider.api_key, 'test-openai-key-12345')

    def test_openai_endpoint_configuration(self):
        """Test OpenAI endpoint configuration"""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        os.environ['OPENAI_ENDPOINT'] = 'https://custom.api.com/v1'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(os.environ['OPENAI_API_KEY'])
            self.assertEqual(provider.base_url, 'https://custom.api.com/v1')

    def test_openai_endpoint_default(self):
        """Test OpenAI endpoint default configuration"""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        # Don't set OPENAI_ENDPOINT, should use default

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(os.environ['OPENAI_API_KEY'])
            self.assertEqual(provider.base_url, 'https://api.openai.com/v1')

    def test_ai_timeout_configuration(self):
        """Test AI timeout configuration"""
        os.environ['AI_TIMEOUT'] = '120'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key")
            self.assertEqual(provider.timeout, 120)

    def test_ai_timeout_default(self):
        """Test AI timeout default configuration"""
        # Don't set AI_TIMEOUT, should use default
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key")
            self.assertEqual(provider.timeout, 60)  # Default timeout

    def test_ai_max_retries_configuration(self):
        """Test AI max retries configuration"""
        os.environ['AI_MAX_RETRIES'] = '5'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key")
            self.assertEqual(provider.max_retries, 5)

    def test_ai_max_retries_default(self):
        """Test AI max retries default configuration"""
        # Don't set AI_MAX_RETRIES, should use default
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key")
            self.assertEqual(provider.max_retries, 3)  # Default max retries

    def test_ai_model_configuration(self):
        """Test AI model configuration"""
        os.environ['AI_MODEL'] = 'custom-test-model'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key", base_url="localhost")
            self.assertEqual(provider.model, 'custom-test-model')

    def test_ai_model_default(self):
        """Test AI model default configuration"""
        # Don't set AI_MODEL, should use default based on provider
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key", base_url="localhost")
            self.assertEqual(provider.model, 'gpt-4-vision-preview')  # Default for localhost

    def test_ai_endpoint_configuration(self):
        """Test AI endpoint configuration"""
        os.environ['AI_ENDPOINT'] = 'https://ai-endpoint.com/v1'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key", base_url=None)
            self.assertEqual(provider.base_url, 'https://ai-endpoint.com/v1')

    def test_environment_variable_precedence(self):
        """Test environment variable precedence"""
        # Set multiple environment variables
        os.environ['OPENAI_API_KEY'] = 'env-openai-key'
        os.environ['AI_MODEL'] = 'env-model'
        os.environ['OPENAI_ENDPOINT'] = 'env-endpoint'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Test that constructor parameters take precedence over environment
            provider = OpenAICompatibleConnector(
                api_key="constructor-key",
                base_url="constructor-endpoint",
                model="constructor-model"
            )

            self.assertEqual(provider.api_key, "constructor-key")
            self.assertEqual(provider.base_url, "constructor-endpoint")
            self.assertEqual(provider.model, "constructor-model")

    def test_environment_variable_fallback(self):
        """Test environment variable fallback when constructor parameters are None"""
        os.environ['OPENAI_API_KEY'] = 'env-openai-key'
        os.environ['OPENAI_ENDPOINT'] = 'env-endpoint'
        os.environ['AI_MODEL'] = 'env-model'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Pass None as constructor parameters, should fall back to environment
            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None,
                model=None
            )

            self.assertEqual(provider.api_key, "env-openai-key")
            self.assertEqual(provider.base_url, "env-endpoint")
            self.assertEqual(provider.model, "env-model")

    def test_empty_environment_variables(self):
        """Test handling of empty environment variables"""
        os.environ['OPENAI_API_KEY'] = ''
        os.environ['OPENAI_ENDPOINT'] = ''
        os.environ['AI_MODEL'] = ''

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None,
                model=None
            )

            # Should use defaults when environment variables are empty
            self.assertEqual(provider.base_url, "https://api.openai.com/v1")
            self.assertEqual(provider.model, "gpt-4o")

    def test_malformed_environment_variables(self):
        """Test handling of malformed environment variables"""
        os.environ['AI_TIMEOUT'] = 'not-a-number'
        os.environ['AI_MAX_RETRIES'] = 'invalid'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector("test-key")

            # Should use defaults when environment variables are malformed
            self.assertEqual(provider.timeout, 60)
            self.assertEqual(provider.max_retries, 3)

    @patch('backend.server.ai_apis')
    @patch('backend.server.logger')
    def test_initialize_ai_apis_with_all_keys(self, mock_logger, mock_ai_apis):
        """Test initialize_ai_apis with all API keys configured"""
        # Set all environment variables
        os.environ['GEMINI_API_KEY'] = 'gemini-key'
        os.environ['OPENROUTER_API_KEY'] = 'openrouter-key'
        os.environ['NVIDIA_API_KEY'] = 'nvidia-key'
        os.environ['OPENAI_API_KEY'] = 'openai-key'

        with patch.dict('sys.modules', {
            'backend.ai_apis.gemini_api': Mock(),
            'backend.ai_apis.openrouter_api': Mock(),
            'backend.ai_apis.nvidia_api': Mock(),
            'backend.ai_apis.openai_compatible': Mock(),
        }):
            # Mock the provider classes
            from unittest.mock import MagicMock
            mock_gemini = MagicMock()
            mock_openrouter = MagicMock()
            mock_nvidia = MagicMock()
            mock_openai = MagicMock()

            with patch('backend.ai_apis.gemini_api.GeminiAPIConnector', mock_gemini), \
                 patch('backend.ai_apis.openrouter_api.OpenRouterAPIConnector', mock_openrouter), \
                 patch('backend.ai_apis.nvidia_api.NVIDIAAPIConnector', mock_nvidia), \
                 patch('backend.ai_apis.openai_compatible.OpenAICompatibleConnector', mock_openai):

                initialize_ai_apis()

                # Verify all providers were initialized
                mock_gemini.assert_called_once_with('gemini-key')
                mock_openrouter.assert_called_once_with('openrouter-key')
                mock_nvidia.assert_called_once_with('nvidia-key')
                mock_openai.assert_called_once_with('openai-key', base_url=None)

    @patch('backend.server.logger')
    def test_initialize_ai_apis_with_no_keys(self, mock_logger):
        """Test initialize_ai_apis with no API keys configured"""
        # Don't set any environment variables

        with patch.dict('sys.modules', {
            'backend.ai_apis.gemini_api': Mock(),
            'backend.ai_apis.openrouter_api': Mock(),
            'backend.ai_apis.nvidia_api': Mock(),
            'backend.ai_apis.openai_compatible': Mock(),
        }):
            from unittest.mock import MagicMock
            mock_gemini = MagicMock()
            mock_openrouter = MagicMock()
            mock_nvidia = MagicMock()
            mock_openai = MagicMock()

            with patch('backend.ai_apis.gemini_api.GeminiAPIConnector', mock_gemini), \
                 patch('backend.ai_apis.openrouter_api.OpenRouterAPIConnector', mock_openrouter), \
                 patch('backend.ai_apis.nvidia_api.NVIDIAAPIConnector', mock_nvidia), \
                 patch('backend.ai_apis.openai_compatible.OpenAICompatibleConnector', mock_openai):

                initialize_ai_apis()

                # Verify no providers were initialized
                mock_gemini.assert_not_called()
                mock_openrouter.assert_not_called()
                mock_nvidia.assert_not_called()
                mock_openai.assert_not_called()

                # Verify warning was logged
                mock_logger.warning.assert_called()

    @patch('backend.server.logger')
    def test_initialize_ai_apis_partial_configuration(self, mock_logger):
        """Test initialize_ai_apis with partial API key configuration"""
        # Set only some environment variables
        os.environ['GEMINI_API_KEY'] = 'gemini-key'
        os.environ['OPENAI_API_KEY'] = 'openai-key'
        # Don't set OPENROUTER_API_KEY or NVIDIA_API_KEY

        with patch.dict('sys.modules', {
            'backend.ai_apis.gemini_api': Mock(),
            'backend.ai_apis.openrouter_api': Mock(),
            'backend.ai_apis.nvidia_api': Mock(),
            'backend.ai_apis.openai_compatible': Mock(),
        }):
            from unittest.mock import MagicMock
            mock_gemini = MagicMock()
            mock_openrouter = MagicMock()
            mock_nvidia = MagicMock()
            mock_openai = MagicMock()

            with patch('backend.ai_apis.gemini_api.GeminiAPIConnector', mock_gemini), \
                 patch('backend.ai_apis.openrouter_api.OpenRouterAPIConnector', mock_openrouter), \
                 patch('backend.ai_apis.nvidia_api.NVIDIAAPIConnector', mock_nvidia), \
                 patch('backend.ai_apis.openai_compatible.OpenAICompatibleConnector', mock_openai):

                initialize_ai_apis()

                # Verify only configured providers were initialized
                mock_gemini.assert_called_once_with('gemini-key')
                mock_openrouter.assert_not_called()
                mock_nvidia.assert_not_called()
                mock_openai.assert_called_once_with('openai-key', base_url=None)

    def test_environment_variable_whitespace_handling(self):
        """Test handling of whitespace in environment variables"""
        os.environ['OPENAI_API_KEY'] = '  trimmed-key  '
        os.environ['OPENAI_ENDPOINT'] = '  https://trimmed.com/v1  '
        os.environ['AI_MODEL'] = '  trimmed-model  '

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None,
                model=None
            )

            # Should trim whitespace
            self.assertEqual(provider.api_key, 'trimmed-key')
            self.assertEqual(provider.base_url, 'https://trimmed.com/v1')
            self.assertEqual(provider.model, 'trimmed-model')

    def test_environment_variable_case_sensitivity(self):
        """Test environment variable case sensitivity"""
        os.environ['openai_api_key'] = 'lowercase-key'  # lowercase
        os.environ['OPENAI_MODEL'] = 'uppercase-model'  # uppercase

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None,
                model=None
            )

            # Environment variables should be case sensitive
            self.assertNotEqual(provider.api_key, 'lowercase-key')
            self.assertEqual(provider.model, 'uppercase-model')

    def test_special_characters_in_environment_variables(self):
        """Test handling of special characters in environment variables"""
        os.environ['OPENAI_API_KEY'] = 'sk-test-key-with-dashes-and_underscores'
        os.environ['OPENAI_ENDPOINT'] = 'https://api.test.com/v1?param=value&other=123'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None
            )

            # Should handle special characters correctly
            self.assertEqual(provider.api_key, 'sk-test-key-with-dashes-and_underscores')
            self.assertEqual(provider.base_url, 'https://api.test.com/v1?param=value&other=123')

    def test_lm_studio_configuration(self):
        """Test LM Studio configuration through environment variables"""
        os.environ['OPENAI_API_KEY'] = 'not-needed'
        os.environ['AI_ENDPOINT'] = 'lm-studio'
        os.environ['AI_MODEL'] = 'local-model'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None,
                model=None
            )

            # Should convert lm-studio to proper endpoint
            self.assertEqual(provider.base_url, 'http://lm-studio:1234/v1')
            self.assertEqual(provider.model, 'local-model')

    def test_localhost_configuration(self):
        """Test localhost configuration through environment variables"""
        os.environ['OPENAI_API_KEY'] = 'not-needed'
        os.environ['AI_ENDPOINT'] = 'localhost'

        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider = OpenAICompatibleConnector(
                api_key=None,
                base_url=None
            )

            # Should convert localhost to proper endpoint
            self.assertEqual(provider.base_url, 'http://localhost:1234/v1')

    def test_environment_variable_validation(self):
        """Test environment variable validation"""
        # Test invalid timeout values
        invalid_values = ['0', '-1', '999999', 'abc', '1.5.3']

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                os.environ['AI_TIMEOUT'] = invalid_value

                with patch('openai.OpenAI') as mock_openai:
                    mock_client = Mock()
                    mock_openai.return_value = mock_client

                    provider = OpenAICompatibleConnector("test-key")
                    # Should fall back to default for invalid values
                    self.assertEqual(provider.timeout, 60)

    def test_environment_variable_combinations(self):
        """Test various combinations of environment variables"""
        test_combinations = [
            {
                'env_vars': {
                    'OPENAI_API_KEY': 'test-key',
                    'AI_TIMEOUT': '90',
                    'AI_MODEL': 'test-model'
                },
                'expected': {
                    'api_key': 'test-key',
                    'timeout': 90,
                    'model': 'test-model'
                }
            },
            {
                'env_vars': {
                    'GEMINI_API_KEY': 'gemini-test',
                    'AI_MAX_RETRIES': '2'
                },
                'expected': {
                    'gemini_key': 'gemini-test',
                    'max_retries': 2
                }
            },
            {
                'env_vars': {
                    'NVIDIA_API_KEY': 'nvidia-test',
                    'NVIDIA_MODEL': 'nvidia/llama3-llm-8b'
                },
                'expected': {
                    'nvidia_key': 'nvidia-test',
                    'nvidia_model': 'nvidia/llama3-llm-8b'
                }
            }
        ]

        for i, combination in enumerate(test_combinations):
            with self.subTest(combination=i):
                # Clear environment
                for var in os.environ.copy():
                    if var.startswith(('GEMINI_', 'OPENROUTER_', 'NVIDIA_', 'OPENAI_', 'AI_')):
                        os.environ.pop(var, None)

                # Set test environment variables
                for var, value in combination['env_vars'].items():
                    os.environ[var] = value

                # Test the expected behavior
                if 'OPENAI_API_KEY' in combination['env_vars']:
                    with patch('openai.OpenAI') as mock_openai:
                        mock_client = Mock()
                        mock_openai.return_value = mock_client

                        provider = OpenAICompatibleConnector(
                            api_key=None,
                            base_url=None,
                            model=None
                        )

                        expected = combination['expected']
                        self.assertEqual(provider.api_key, expected['api_key'])
                        self.assertEqual(provider.timeout, expected['timeout'])
                        self.assertEqual(provider.model, expected['model'])

                elif 'GEMINI_API_KEY' in combination['env_vars']:
                    with patch('requests.post') as mock_post:
                        mock_response = Mock()
                        mock_response.json.return_value = {
                            "candidates": [{"content": {"parts": [{"text": "UP"}]}}]
                        }
                        mock_response.raise_for_status.return_value = None
                        mock_post.return_value = mock_response

                        provider = GeminiAPIConnector(None)
                        expected = combination['expected']
                        self.assertEqual(provider.api_key, expected['gemini_key'])
                        self.assertEqual(provider.max_retries, expected['max_retries'])

                elif 'NVIDIA_API_KEY' in combination['env_vars']:
                    with patch('requests.post') as mock_post:
                        mock_response = Mock()
                        mock_response.json.return_value = {
                            "choices": [{"message": {"content": "A"}}]
                        }
                        mock_response.raise_for_status.return_value = None
                        mock_post.return_value = mock_response

                        provider = NVIDIAAPIConnector(None)
                        expected = combination['expected']
                        self.assertEqual(provider.api_key, expected['nvidia_key'])
                        self.assertEqual(provider.model, expected['nvidia_model'])


if __name__ == '__main__':
    unittest.main(verbosity=2)