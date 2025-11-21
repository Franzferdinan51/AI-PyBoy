"""
NVIDIA NIM AI API connector
"""
from openai import OpenAI
import base64
import os
import requests
from typing import List, Optional, Dict, Any
import io
from PIL import Image
from .ai_api_base import AIAPIConnector


class NVIDIAAPIConnector(AIAPIConnector):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # NVIDIA uses OpenAI-compatible API
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = None
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        self.timeout = int(os.environ.get('AI_TIMEOUT', '60'))
        self.max_retries = int(os.environ.get('AI_MAX_RETRIES', '3'))

        # Initialize client with robust error handling
        self.client = self._initialize_client()

        # Test connection during initialization
        self._test_connection()

    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client for NVIDIA API with error handling"""
        try:
            if not self.api_key:
                self.logger.warning("No API key provided for NVIDIA API")
                return None

            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            self.logger.info(f"Initialized NVIDIA API client with URL: {self.base_url}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize NVIDIA API client: {e}")
            return None

    def _test_connection(self):
        """Test connection to the NVIDIA API"""
        if not self.client:
            self.logger.warning("Cannot test connection - no client initialized")
            return

        try:
            # Simple test request
            response = self.client.models.list()
            self.logger.info(f"Connection test successful. Available models: {len(response.data) if hasattr(response, 'data') else 'unknown'}")
        except Exception as e:
            self.logger.warning(f"Connection test failed: {e}")
            # Don't raise exception - allow the connector to work even if test fails

    def get_models(self) -> List[str]:
        """Get a list of available models from the NVIDIA API"""
        if not self.client:
            self.logger.warning("Cannot fetch models - client not initialized")
            return []

        try:
            response = self.client.models.list()
            models = [model.id for model in response.data]
            self.logger.info(f"Found {len(models)} models at {self.base_url}")
            return models
        except Exception as e:
            self.logger.error(f"Failed to fetch models from {self.base_url}: {e}")
            return []

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str]) -> str:
        """Get the next action from NVIDIA API based on the current game state"""
        # Check if client is initialized
        if not self.client:
            self.logger.error("NVIDIA API client not initialized")
            return self._get_fallback_action(action_history)

        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Create enhanced prompt with better context
            prompt = self._create_action_prompt(goal, action_history)

            self.logger.debug(f"Making request to NVIDIA API - Model: {self.model}, URL: {self.base_url}")

            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else "meta/llama3-8b-instruct"

            # Create request with retry logic
            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=10,
                    temperature=0.7,
                    timeout=self.timeout
                )

            response = self._retry_with_backoff(make_request)

            # Parse and validate response
            action = self._parse_action_response(response)
            if action in self.valid_actions:
                self.logger.info(f"NVIDIA API returned valid action: {action}")
                return action
            else:
                self.logger.warning(f"NVIDIA API returned invalid action: '{action}'. Using fallback.")
                return self._get_fallback_action(action_history)

        except Exception as e:
            self.logger.error(f"Error calling NVIDIA API: {e}", exc_info=True)
            return self._get_fallback_action(action_history)

    def _create_action_prompt(self, goal: str, action_history: List[str]) -> str:
        """Create an enhanced prompt for action selection"""
        recent_actions = ', '.join(action_history[-5:]) if action_history else "none"
        return f"""You are an expert AI playing a retro video game. Your current high-level objective is: "{goal}".

Recent actions taken: {recent_actions}.

Based on the provided game screenshot, determine the single next button press to advance toward the objective.

Your response MUST be one of the following exact words: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
Do not provide any explanation or other text."""

    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI"""
        return "You are an expert AI playing a retro video game. Respond with only the action name in uppercase."

    def _parse_action_response(self, response) -> str:
        """Parse and clean action response from AI"""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content.strip().upper()
                # Clean up common response issues
                content = content.replace('.', '').replace(',', '').strip()
                return content
            return "SELECT"
        except Exception as e:
            self.logger.error(f"Error parsing action response: {e}")
            return "SELECT"

    def _get_fallback_action(self, action_history: List[str]) -> str:
        """Get a fallback action when AI fails"""
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

    def chat_with_ai(self, user_message: str, image_bytes: bytes, context: dict) -> str:
        """Chat with the AI about the current game state"""
        if not self.client:
            return "I'm sorry, the AI service is not available right now."

        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            prompt = self._create_chat_prompt(user_message, context)

            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else "meta/llama3-8b-instruct"

            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful game assistant with expertise in retro video games."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    timeout=self.timeout
                )

            response = self._retry_with_backoff(make_request)
            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error in AI chat: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request. Please try again."