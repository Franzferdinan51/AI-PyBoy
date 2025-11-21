# AI Game Server

A unified backend server that combines PyBoy and PyGBA emulators with multiple AI APIs to create an AI-powered game playing system similar to "GPT plays Pokémon".

## Features

- Supports both Game Boy (PyBoy) and Game Boy Advance (PyGBA) emulation
- Integrates with multiple AI APIs:
  - Google Gemini
  - OpenRouter (supports various models including GPT-4 Vision)
  - NVIDIA NIM
  - OpenAI Compatible APIs (LM Studio, Ollama, etc.)
- RESTful API for controlling the emulators and getting AI decisions
- Action history tracking
- State management (save/load states)
- Dynamic model selection for all providers
- Automatic fallback between providers

## Prerequisites

- Python 3.8+
- PyBoy (for Game Boy emulation)
- PyGBA (for Game Boy Advance emulation, optional)
- API keys for the AI services you want to use

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-game-server
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install emulator packages:
   ```bash
   pip install pyboy
   pip install pygba  # Optional, for GBA support
   ```

## Configuration

Set the following environment variables for the AI APIs you want to use:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export NVIDIA_API_KEY="your-nvidia-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # For LM Studio and other OpenAI-compatible APIs
```

## AI Provider Requirements and Capabilities

### Google Gemini

**Vision Capabilities**: ✅ Required
**Tool Use**: ❌ Not Required
**Model Requirements**: 
- Must support vision/image analysis
- Recommended models: `gemini-1.5-pro`, `gemini-1.5-flash`

**Setup**:
1. Get an API key from [Google AI Studio](https://aistudio.google.com/)
2. Set `GEMINI_API_KEY` environment variable

**Features**:
- Excellent at understanding game screenshots
- Good reasoning capabilities for complex objectives
- Fast response times

### OpenRouter

**Vision Capabilities**: ✅ Required (for most models)
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis
- Supports a wide variety of models including:
  - OpenAI GPT-4 Vision (`openai/gpt-4-vision-preview`)
  - Anthropic Claude (`anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`)
  - Google Gemini (`google/gemini-pro-vision`)
  - Mistral (`mistralai/mistral-large`)
  - And many more...

**Setup**:
1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Get an API key from your account settings
3. Set `OPENROUTER_API_KEY` environment variable

**Features**:
- Access to a wide variety of cutting-edge models
- Competitive pricing
- No rate limits for most models
- Easy model switching

**Model Format**: 
Models must be specified in the format `vendor/model-name`. Examples:
- `openai/gpt-4-vision-preview`
- `anthropic/claude-3-opus:beta`
- `google/gemini-pro-vision`
- `mistralai/mistral-large`

### NVIDIA NIM

**Vision Capabilities**: ✅ Required
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis
- Available models include:
  - `meta/llama3-8b-instruct`
  - `meta/llama3-70b-instruct`
  - `mistralai/mistral-7b-instruct-v0.2`
  - `mistralai/mixtral-8x7b-instruct-v0.1`
  - And others...

**Setup**:
1. Access NVIDIA NIM through [NVIDIA API Catalog](https://build.nvidia.com/)
2. Get an API key
3. Set `NVIDIA_API_KEY` environment variable

**Features**:
- High-performance models optimized by NVIDIA
- Fast inference times
- Access to Meta's Llama models
- Enterprise-grade reliability

### OpenAI Compatible APIs (LM Studio, Ollama, etc.)

**Vision Capabilities**: ✅ Required (for local vision models)
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis if running locally
- Can use any OpenAI-compatible model
- Popular local models:
  - `llava:13b` (vision capable)
  - `bakllava` (vision capable)
  - `gpt-4-vision-preview` (if proxied)

**Setup for LM Studio**:
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a vision-capable model (e.g., LLaVA)
3. Start the local server in LM Studio
4. The default endpoint is `http://localhost:1234/v1`
5. API key is typically not required for local use

**Setup for Ollama**:
1. Install [Ollama](https://ollama.com/)
2. Pull a vision model: `ollama pull llava`
3. Run the model: `ollama run llava`
4. The endpoint will be `http://localhost:11434/v1`

**Features**:
- Complete privacy (no data leaves your machine)
- No API costs
- Full control over model selection
- Requires sufficient local hardware (GPU recommended)

**Note**: For local providers, the API key field can typically be left blank.

## Usage

1. Start the server:
   ```bash
   python src/main.py
   ```

2. The server will start on `http://localhost:5000`

## API Endpoints

### GET /api/status
Get the current status of the server and available AI providers.

### POST /api/load-rom
Load a ROM file into the specified emulator.
```json
{
  "rom_path": "/path/to/rom/file",
  "emulator_type": "gb"  // or "gba"
}
```

### GET /api/screen
Get the current screen from the active emulator as a base64 encoded JPEG image.

### POST /api/action
Execute an action in the active emulator.
```json
{
  "action": "A",  // UP, DOWN, LEFT, RIGHT, A, B, START, SELECT
  "frames": 1
}
```

### POST /api/ai-action
Get the next action from the specified AI API.
```json
{
  "api_name": "gemini",  // gemini, openrouter, nvidia, openai-compatible
  "api_endpoint": "http://localhost:1234/v1",  // Optional, for local providers
  "api_key": "your-api-key",  // Optional, if not set in environment
  "model": "models/gemini-1.5-pro-latest",  // Optional, model selection
  "goal": "Defeat the first gym leader"
}
```

### GET /api/models?provider={provider_name}
Get a list of available models for the specified provider.
Example: `GET /api/models?provider=gemini`

### POST /api/reset
Reset the active emulator.

### GET /api/info
Get information about the current game state.

## Example Workflow

1. Load a ROM:
   ```bash
   curl -X POST http://localhost:5000/api/load-rom \
        -H "Content-Type: application/json" \
        -d '{"rom_path": "/path/to/pokemon.gb", "emulator_type": "gb"}'
   ```

2. Get the current screen:
   ```bash
   curl http://localhost:5000/api/screen
   ```

3. Get an AI action:
   ```bash
   curl -X POST http://localhost:5000/api/ai-action \
        -H "Content-Type: application/json" \
        -d '{"api_name": "gemini", "goal": "Get out of the house and go to the lab"}'
   ```

4. Execute the action:
   ```bash
   curl -X POST http://localhost:5000/api/action \
        -H "Content-Type: application/json" \
        -d '{"action": "DOWN", "frames": 10}'
   ```

## Frontend UI

The project includes a React-based frontend that provides:
- ROM loading interface
- Real-time game screen display
- AI control panel with goal setting
- Action history tracking
- Provider and model selection
- Chat interface for interacting with the AI

To run the frontend:
1. Navigate to the `ai-game-assistant` directory
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

## Development

### Project Structure
```
ai-game-server/
├── src/
│   ├── backend/
│   │   ├── emulators/
│   │   │   ├── emulator_interface.py
│   │   │   ├── pyboy_emulator.py
│   │   │   └── pygba_emulator.py
│   │   ├── ai_apis/
│   │   │   ├── ai_api_base.py
│   │   │   ├── ai_provider_manager.py
│   │   │   ├── gemini_api.py
│   │   │   ├── openrouter_api.py
│   │   │   ├── nvidia_api.py
│   │   │   └── openai_compatible.py
│   │   └── server.py
│   └── main.py
├── requirements.txt
└── README.md
```

### Adding New AI APIs

1. Create a new connector class in `src/backend/ai_apis/` that inherits from `AIAPIConnector`
2. Implement the required methods:
   - `get_next_action`: Returns the next game action
   - `chat_with_ai`: Handles chat interactions
   - `get_models`: Returns available models (optional but recommended)
3. Add initialization code in `ai_provider_manager.py` to register the new API

### Adding New Emulators

1. Create a new emulator class in `src/backend/emulators/` that implements `EmulatorInterface`
2. Implement all the required methods:
   - `load_rom`: Load a ROM file
   - `get_screen`: Get the current screen as a numpy array
   - `step`: Execute an action for a number of frames
3. Register the new emulator in `server.py`

## Troubleshooting

### Common Issues

1. **No AI providers available**: Make sure you've set the appropriate environment variables with your API keys.

2. **Model not found**: For OpenRouter and NVIDIA, make sure you're using the correct model identifier format.

3. **Local provider connection issues**: 
   - Ensure your local server (LM Studio, Ollama, etc.) is running
   - Check that the endpoint URL is correct
   - Verify that the model is loaded in your local server

4. **Vision model required**: All providers must use vision-capable models since the AI needs to analyze game screenshots.

### Performance Tips

1. **For best results**: Use vision-capable models with strong reasoning abilities
2. **For fastest responses**: Local providers (LM Studio, Ollama) have no network latency
3. **For strongest AI**: Cloud providers like Gemini and OpenRouter with top-tier models
4. **Cost considerations**: Local providers are free but require powerful hardware

## License

This project is licensed under the MIT License.