"""
AI Provider Manager - Handles automatic provider detection and fallback
"""
import os
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from .ai_api_base import AIAPIConnector
from .gemini_api import GeminiAPIConnector
from .openrouter_api import OpenRouterAPIConnector
from .openai_compatible import OpenAICompatibleConnector
from .nvidia_api import NVIDIAAPIConnector

class ProviderStatus(Enum):
    """Provider status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"

class AIProviderManager:
    """Manages AI providers with automatic detection and fallback"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.provider_order: List[str] = []
        self.fallback_providers: List[str] = []
        self.initialize_providers()

    def initialize_providers(self):
        """Initialize all available AI providers"""
        self.logger.info("Initializing AI providers...")

        # Define provider configurations
        provider_configs = [
            {
                'name': 'gemini',
                'env_key': 'GEMINI_API_KEY',
                'class': GeminiAPIConnector,
                'priority': 1
            },
            {
                'name': 'openrouter',
                'env_key': 'OPENROUTER_API_KEY',
                'class': OpenRouterAPIConnector,
                'priority': 2
            },
            {
                'name': 'openai-compatible',
                'env_key': 'OPENAI_API_KEY',
                'class': OpenAICompatibleConnector,
                'priority': 3,
                'extra_params': {
                    'base_url': os.environ.get('OPENAI_ENDPOINT')
                }
            },
            {
                'name': 'nvidia',
                'env_key': 'NVIDIA_API_KEY',
                'class': NVIDIAAPIConnector,
                'priority': 4
            }
        ]

        # Initialize providers
        for config in provider_configs:
            self._initialize_provider(config)

        # Sort providers by priority
        self.provider_order = sorted(
            [name for name, info in self.providers.items() if info['status'] == ProviderStatus.AVAILABLE],
            key=lambda x: self.providers[x]['priority']
        )

        # Set up fallback providers
        self.fallback_providers = self.provider_order.copy()
        self.logger.info(f"Provider initialization complete. Available providers: {self.provider_order}")

    def _initialize_provider(self, config: Dict[str, Any]):
        """Initialize a single provider"""
        name = config['name']
        env_key = config['env_key']
        provider_class = config['class']
        priority = config['priority']
        extra_params = config.get('extra_params', {})

        try:
            api_key = os.environ.get(env_key)
            if not api_key:
                # For local providers, API key might not be required
                if name == 'openai-compatible' and extra_params.get('base_url'):
                    if 'localhost' in extra_params['base_url'] or '127.0.0.1' in extra_params['base_url']:
                        api_key = "not-needed"  # Local provider
                    else:
                        self.logger.info(f"API key not found for {name} (environment variable: {env_key})")
                        self.providers[name] = {
                            'status': ProviderStatus.UNAVAILABLE,
                            'connector': None,
                            'priority': priority,
                            'error': f"API key not found in environment variable: {env_key}"
                        }
                        return
                else:
                    self.logger.info(f"API key not found for {name} (environment variable: {env_key})")
                    self.providers[name] = {
                        'status': ProviderStatus.UNAVAILABLE,
                        'connector': None,
                        'priority': priority,
                        'error': f"API key not found in environment variable: {env_key}"
                    }
                    return

            # Initialize the connector
            if extra_params:
                connector = provider_class(api_key, **extra_params)
            else:
                connector = provider_class(api_key)

            # Test the connection
            self.providers[name] = {
                'status': ProviderStatus.AVAILABLE,
                'connector': connector,
                'priority': priority,
                'error': None
            }

            self.logger.info(f"Successfully initialized {name} provider")

        except Exception as e:
            self.logger.error(f"Failed to initialize {name} provider: {e}", exc_info=True)
            self.providers[name] = {
                'status': ProviderStatus.ERROR,
                'connector': None,
                'priority': priority,
                'error': str(e)
            }

    def get_provider(self, provider_name: Optional[str] = None) -> Optional[AIAPIConnector]:
        """Get a provider connector by name, or use the first available one"""
        if provider_name:
            # Try specific provider first
            if provider_name in self.providers:
                provider_info = self.providers[provider_name]
                if provider_info['status'] == ProviderStatus.AVAILABLE:
                    return provider_info['connector']
                else:
                    self.logger.warning(f"Provider {provider_name} is not available: {provider_info.get('error', 'Unknown error')}")
            else:
                self.logger.warning(f"Unknown provider: {provider_name}")

        # Fall back to automatic provider selection
        for provider_name in self.provider_order:
            provider_info = self.providers[provider_name]
            if provider_info['status'] == ProviderStatus.AVAILABLE:
                self.logger.debug(f"Using provider: {provider_name}")
                return provider_info['connector']

        self.logger.error("No available AI providers found")
        return None

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str],
                      provider_name: Optional[str] = None, model: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Get next action with automatic fallback"""
        # Try specified provider first
        if provider_name:
            connector = self.get_provider(provider_name)
            if connector:
                try:
                    if model:
                        connector.model = model
                    action = connector.get_next_action(image_bytes, goal, action_history)
                    return action, provider_name
                except Exception as e:
                    self.logger.error(f"Provider {provider_name} failed: {e}")
                    # Continue to fallback
            else:
                self.logger.warning(f"Provider {provider_name} not available, falling back")

        # Try providers in order
        for fallback_provider in self.fallback_providers:
            connector = self.get_provider(fallback_provider)
            if connector:
                try:
                    # Set model if provided
                    if model:
                        connector.model = model
                    action = connector.get_next_action(image_bytes, goal, action_history)
                    self.logger.info(f"Successfully used fallback provider: {fallback_provider}")
                    return action, fallback_provider
                except Exception as e:
                    self.logger.error(f"Fallback provider {fallback_provider} failed: {e}")
                    continue

        # Ultimate fallback - use a default action
        self.logger.error("All providers failed, using default action")
        return self._get_default_action(action_history), None

    def chat_with_ai(self, message: str, image_bytes: bytes, context: dict,
                    provider_name: Optional[str] = None, model: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Chat with AI with automatic fallback"""
        # Try specified provider first
        if provider_name:
            connector = self.get_provider(provider_name)
            if connector:
                try:
                    if model:
                        connector.model = model
                    response = connector.chat_with_ai(message, image_bytes, context)
                    return response, provider_name
                except Exception as e:
                    self.logger.error(f"Provider {provider_name} failed: {e}")
                    # Continue to fallback
            else:
                self.logger.warning(f"Provider {provider_name} not available, falling back")

        # Try providers in order
        for fallback_provider in self.fallback_providers:
            connector = self.get_provider(fallback_provider)
            if connector:
                try:
                    # Set model if provided
                    if model:
                        connector.model = model
                    response = connector.chat_with_ai(message, image_bytes, context)
                    self.logger.info(f"Successfully used fallback provider for chat: {fallback_provider}")
                    return response, fallback_provider
                except Exception as e:
                    self.logger.error(f"Fallback provider {fallback_provider} failed: {e}")
                    continue

        # Ultimate fallback
        return "I'm sorry, all AI services are currently unavailable. Please try again later.", None

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for name, info in self.providers.items():
            status[name] = {
                'status': info['status'].value,
                'priority': info['priority'],
                'error': info.get('error'),
                'available': info['status'] == ProviderStatus.AVAILABLE
            }
        return status

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, info in self.providers.items() if info['status'] == ProviderStatus.AVAILABLE]

    def get_models(self, provider_name: str) -> List[str]:
        """Get a list of available models for a given provider"""
        provider = self.get_provider(provider_name)
        if provider:
            try:
                return provider.get_models()
            except Exception as e:
                self.logger.error(f"Failed to get models for {provider_name}: {e}")
                return []
        return []

    def _get_default_action(self, action_history: List[str]) -> str:
        """Get a default action when all providers fail"""
        # Simple strategy: if no recent actions, try UP, otherwise try different action
        if not action_history:
            return "UP"

        last_action = action_history[-1]
        if last_action == "UP":
            return "RIGHT"
        elif last_action == "RIGHT":
            return "DOWN"
        elif last_action == "DOWN":
            return "LEFT"
        else:
            return "A"

    def refresh_provider_status(self):
        """Refresh the status of all providers"""
        self.logger.info("Refreshing provider status...")
        for name, info in self.providers.items():
            if info['connector']:
                try:
                    # Simple test - try to access a basic property
                    if hasattr(info['connector'], 'client'):
                        if info['connector'].client:
                            info['status'] = ProviderStatus.AVAILABLE
                        else:
                            info['status'] = ProviderStatus.UNAVAILABLE
                    else:
                        info['status'] = ProviderStatus.AVAILABLE
                except Exception as e:
                    info['status'] = ProviderStatus.ERROR
                    info['error'] = str(e)
        self.logger.info("Provider status refresh complete")

# Global instance
ai_provider_manager = AIProviderManager()