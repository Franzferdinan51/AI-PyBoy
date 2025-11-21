# AI Provider Setup Guide

This guide explains how to configure and use different AI providers with the Pokemon game automation system.

## Supported AI Providers

The system supports multiple AI providers with automatic fallback capabilities:

### 1. Google Gemini (Recommended for beginners)
- **Free tier available**
- Good vision capabilities
- Easy to set up

### 2. OpenRouter
- **Paid service**
- Access to multiple models including GPT-4 Vision
- Good performance

### 3. OpenAI-Compatible APIs
- **Flexible endpoints**
- Supports OpenAI, LM Studio, Ollama, and other local providers
- Perfect for running local models

### 4. NVIDIA NIM
- **Paid service**
- High-performance models
- Good for production use

## Environment Variable Configuration

### Basic Setup

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

### Provider-Specific Configuration

#### Google Gemini
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**How to get API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy it to your `.env` file

#### OpenRouter
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**How to get API key:**
1. Go to https://openrouter.ai/keys
2. Create an account and deposit funds
3. Generate an API key
4. Copy it to your `.env` file

#### OpenAI-Compatible APIs (LM Studio, Ollama, etc.)

For LM Studio:
```env
OPENAI_API_KEY=not-needed
OPENAI_ENDPOINT=http://localhost:1234/v1
OPENAI_MODEL=local-model
```

For Ollama:
```env
OPENAI_API_KEY=not-needed
OPENAI_ENDPOINT=http://localhost:11434/v1
OPENAI_MODEL=llama3
```

For OpenAI:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

#### NVIDIA NIM
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=nvidia/llama3-llm-70b
```

## Local AI Provider Setup

### LM Studio Setup

1. **Download and install LM Studio** from https://lmstudio.ai/
2. **Download a model**:
   - Go to the search tab (magnifying glass icon)
   - Search for vision-capable models like:
     - `llava-v1.6-34b`
     - `bakllava-v1`
     - Any model with "vision" or "multimodal" in the name
3. **Configure the server**:
   - Go to the speech bubble icon (☁️) on the left
   - Click "Start Server"
   - Set port to 1234 (default)
   - Select your downloaded model
   - Enable "Cross-Origin Resource Sharing (CORS)"
4. **Test the connection**:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### Ollama Setup

1. **Download and install Ollama** from https://ollama.ai/
2. **Download a vision model**:
   ```bash
   ollama pull llava
   # or
   ollama pull bakllava
   ```
3. **Start the server**:
   ```bash
   ollama serve
   ```
4. **Test the connection**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Testing Your Setup

### 1. Check Provider Status
Start the server and visit:
```
http://localhost:5000/api/providers/status
```

This will show you which providers are available and their status.

### 2. Test AI Action Endpoint
```bash
curl -X POST http://localhost:5000/api/ai-action \
  -H "Content-Type: application/json" \
  -d '{
    "api_name": "openai-compatible",
    "goal": "Navigate through the game",
    "api_key": "not-needed",
    "api_endpoint": "http://localhost:1234/v1"
  }'
```

### 3. Test Chat Endpoint
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you see on the screen?",
    "api_name": "openai-compatible",
    "api_key": "not-needed",
    "api_endpoint": "http://localhost:1234/v1"
  }'
```

## Troubleshooting

### Common Issues

1. **"No AI providers available"**
   - Check your API keys in `.env`
   - Verify your local AI server is running
   - Check server logs for detailed error messages

2. **"Connection refused" for local providers**
   - Make sure LM Studio/Ollama is running
   - Verify the port number (1234 for LM Studio, 11434 for Ollama)
   - Check firewall settings

3. **"Model not found" errors**
   - Verify the model name matches what's available in your AI tool
   - For LM Studio, check the server tab for the exact model name
   - For Ollama, run `ollama list` to see available models

4. **API timeout errors**
   - Increase timeout in `.env`: `AI_TIMEOUT=120`
   - Check if your model is too large for your hardware
   - Try a smaller model

### Debug Mode

Enable debug logging in your `.env` file:
```env
LOG_LEVEL=DEBUG
FLASK_DEBUG=true
```

### Performance Tips

1. **Local Models**: Use smaller models for better performance
2. **Cloud APIs**: Start with Gemini (free) before trying paid services
3. **Caching**: The system automatically caches provider connections
4. **Fallback**: Configure multiple providers for automatic failover

## Advanced Configuration

### Custom Provider Endpoints

You can add custom OpenAI-compatible providers by setting:
```env
OPENAI_ENDPOINT=https://your-custom-endpoint.com/v1
OPENAI_MODEL=your-model-name
```

### Provider Priority

The system automatically prioritizes providers in this order:
1. Gemini
2. OpenRouter
3. OpenAI-compatible
4. NVIDIA NIM

### Timeout and Retry Configuration

```env
AI_TIMEOUT=60          # Timeout in seconds
AI_MAX_RETRIES=3       # Number of retry attempts
```

## API Reference

### POST /api/ai-action
Get the next game action from AI

**Request Body:**
```json
{
  "api_name": "optional_provider_name",
  "goal": "your_game_objective",
  "api_key": "your_api_key",
  "api_endpoint": "custom_endpoint_url"
}
```

**Response:**
```json
{
  "action": "UP",
  "provider_used": "gemini",
  "history": ["LEFT", "UP", "A"]
}
```

### POST /api/chat
Chat with the AI about the game

**Request Body:**
```json
{
  "message": "What do you see?",
  "api_name": "optional_provider_name",
  "api_key": "your_api_key",
  "api_endpoint": "custom_endpoint_url"
}
```

**Response:**
```json
{
  "response": "I see a Pokemon game screen...",
  "provider_used": "gemini"
}
```

### GET /api/providers/status
Get status of all AI providers

**Response:**
```json
{
  "gemini": {
    "status": "available",
    "priority": 1,
    "error": null,
    "available": true
  },
  "openai-compatible": {
    "status": "available",
    "priority": 3,
    "error": null,
    "available": true
  }
}
```