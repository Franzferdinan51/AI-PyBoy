"""
Main server application for AI game playing
"""
import os
import io
import json
import base64
import logging
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, send_file, Response
import time
from flask_cors import CORS
import numpy as np
from PIL import Image

# Import configuration
try:
    from ...config import *
except ImportError:
    # Default configuration if config.py is not found
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    DEFAULT_EMULATOR = "gb"
    SCREEN_CAPTURE_FORMAT = "jpeg"
    SCREEN_CAPTURE_QUALITY = 85
    DEFAULT_AI_API = "gemini"
    AI_REQUEST_TIMEOUT = 30
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "{asctime} - {name} - {levelname} - {message}"
    LOG_FILE = "ai_game_server.log"
    MAX_ROM_SIZE = 100 * 1024 * 1024
    ALLOWED_ROM_EXTENSIONS = [".gb", ".gbc", ".gba"]
    ACTION_HISTORY_LIMIT = 1000

# Use absolute imports instead of relative imports
from emulators.pyboy_emulator import PyBoyEmulator
from emulators.pygba_emulator import PyGBAEmulator
from ai_apis.ai_provider_manager import ai_provider_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    style='{',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS with specific settings for frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175", "http://127.0.0.1:5176"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "max_age": 86400,
        "send_wildcard": False
    }
})

# Global emulator instances
emulators = {
    "gb": PyBoyEmulator(),
    "gba": PyGBAEmulator()
}

# AI provider manager is imported from ai_provider_manager

# Action history
action_history = []

# Game state
game_state = {
    "active_emulator": None,
    "rom_loaded": False,
    "ai_running": False,
    "current_goal": ""
}

def numpy_to_base64_image(np_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG image"""
    try:
        # Validate input array
        if np_array is None or np_array.size == 0:
            logger.error("Invalid numpy array: None or empty")
            return ""

        # Ensure array is 3D (height, width, channels)
        if len(np_array.shape) == 2:
            # Grayscale to RGB
            np_array = np.stack([np_array] * 3, axis=2)
        elif len(np_array.shape) != 3:
            logger.error(f"Invalid array shape: {np_array.shape}")
            return ""

        # Ensure RGB format (remove alpha if present)
        if np_array.shape[2] == 4:
            np_array = np_array[:, :, :3]
        elif np_array.shape[2] != 3:
            logger.error(f"Invalid channel count: {np_array.shape[2]}")
            return ""

        # Convert to uint8 if needed
        if np_array.dtype != np.uint8:
            if np_array.dtype in [np.float32, np.float64]:
                # Normalize float values to 0-255 range
                np_array = np.clip(np_array * 255, 0, 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)

        # Create PIL Image
        try:
            image = Image.fromarray(np_array)
        except Exception as e:
            logger.error(f"Failed to create PIL image: {e}")
            return ""

        # Convert to JPEG
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85, optimize=True)
        img_buffer.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        if not img_base64:
            logger.error("Base64 conversion resulted in empty string")
            return ""

        return img_base64

    except Exception as e:
        logger.error(f"Error converting numpy array to base64 image: {e}")
        return ""

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint for monitoring"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get comprehensive status of the server"""
    status = game_state.copy()
    status['ai_providers'] = ai_provider_manager.get_provider_status()
    return jsonify(status), 200

@app.route('/api/providers/status', methods=['GET'])
def get_providers_status():
    """Get detailed status of all AI providers"""
    return jsonify(ai_provider_manager.get_provider_status()), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get a list of available models for a given provider"""
    provider_name = request.args.get('provider')
    if not provider_name:
        return jsonify({"error": "Provider name is required"}), 400

    models = ai_provider_manager.get_models(provider_name)
    return jsonify({"models": models}), 200

@app.route('/api/upload-rom', methods=['POST'])
def upload_rom():
    """Upload a ROM file and load it into the specified emulator"""
    try:
        if 'rom_file' not in request.files:
            return jsonify({"error": "No ROM file provided"}), 400

        file = request.files['rom_file']
        emulator_type = request.form.get('emulator_type', 'gb')
        launch_ui = request.form.get('launch_ui', 'true').lower() == 'true'

        if file.filename == '':
            return jsonify({"error": "No ROM file selected"}), 400

        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_ROM_EXTENSIONS:
            return jsonify({"error": f"Invalid file extension. Allowed: {ALLOWED_ROM_EXTENSIONS}"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_rom_path = temp_file.name

        if emulator_type not in emulators:
            os.unlink(temp_rom_path)
            return jsonify({"error": f"Invalid emulator type. Available: {list(emulators.keys())}"}), 400

        success = emulators[emulator_type].load_rom(temp_rom_path)

        if success:
            game_state["active_emulator"] = emulator_type
            game_state["rom_loaded"] = True
            logger.info(f"Successfully loaded ROM: {file.filename} into {emulator_type} emulator")

            emulator = emulators[emulator_type]
            if hasattr(emulator, 'pyboy') and emulator.pyboy:
                for _ in range(100):
                    emulator.pyboy.tick()

            # UI is now launched automatically by the emulator
            ui_status = emulator.get_ui_status() if hasattr(emulator, 'get_ui_status') else {"running": False}

            response_data = {
                "message": "ROM loaded successfully",
                "rom_name": file.filename,
                "ui_launched": ui_status.get("running", False),
                "ui_status": ui_status
            }

            if not ui_status.get("running", False) and launch_ui:
                logger.warning("UI process failed to launch automatically")

            return jsonify(response_data), 200
        else:
            os.unlink(temp_rom_path)
            return jsonify({"error": "Failed to load ROM"}), 500

    except Exception as e:
        logger.error(f"Error uploading ROM: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/screen', methods=['GET'])
def get_screen():
    """Get the current screen from the active emulator"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        emulator = emulators[game_state["active_emulator"]]
        emulator.step('NOOP', 1)
        screen_array = emulator.get_screen()
        img_base64 = numpy_to_base64_image(screen_array)
        
        return jsonify({
            "image": img_base64,
            "shape": screen_array.shape
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting screen: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/stream', methods=['GET'])
def stream_screen():
    """SSE endpoint for live screen streaming"""
    def generate():
        logger.info("SSE stream requested. Checking game state...")
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("SSE stream aborted: No ROM loaded.")
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            return

        emulator = emulators[game_state["active_emulator"]]
        logger.info(f"Starting SSE stream for {game_state['active_emulator']}.")

        frame_count = 0
        last_frame_time = time.time()
        target_fps = 30  # Reduce to 30 FPS for better stability
        frame_interval = 1.0 / target_fps

        # Send initial frame immediately
        try:
            screen_array = emulator.get_screen()
            img_base64 = numpy_to_base64_image(screen_array)
            initial_data = {
                'image': img_base64,
                'shape': screen_array.shape,
                'timestamp': time.time(),
                'frame': frame_count,
                'fps': target_fps,
                'status': 'stream_started'
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            frame_count += 1
        except Exception as e:
            logger.error(f"Error sending initial frame: {e}")
            yield f"data: {json.dumps({'error': f'Initial frame error: {str(e)}'})}\n\n"
            return

        while True:
            try:
                # Calculate timing for consistent frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time

                # Only process frame if enough time has passed
                if elapsed >= frame_interval:
                    # Step emulator with NOOP action (advance 1 frame)
                    success = emulator.step('NOOP', 1)
                    if not success:
                        logger.warning(f"Stream frame {frame_count}: emulator.step() returned False.")

                    # Get screen after stepping
                    screen_array = emulator.get_screen()

                    # Validate screen array before processing
                    if screen_array is None or screen_array.size == 0:
                        logger.warning(f"Stream frame {frame_count}: Empty screen array")
                        continue

                    img_base64 = numpy_to_base64_image(screen_array)

                    if not img_base64:
                        logger.warning(f"Stream frame {frame_count}: Failed to convert screen to base64")
                        continue

                    data = {
                        'image': img_base64,
                        'shape': screen_array.shape,
                        'timestamp': current_time,
                        'frame': frame_count,
                        'fps': target_fps,
                        'actual_interval': elapsed
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                    if frame_count % 60 == 0:  # Log every 60 frames
                        logger.debug(f"Streamed frame {frame_count} successfully. Interval: {elapsed:.3f}s")

                    frame_count += 1
                    last_frame_time = current_time
                else:
                    # Sleep for the remaining time to maintain frame rate
                    remaining_time = frame_interval - elapsed
                    if remaining_time > 0:
                        time.sleep(remaining_time * 0.8)  # Sleep slightly less to account for processing time

            except GeneratorExit:
                logger.info("SSE stream client disconnected.")
                break
            except Exception as e:
                logger.error(f"Error in SSE stream loop at frame {frame_count}: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

    response = Response(generate(), mimetype='text/event-stream')
    # Add SSE-specific headers
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')  # Disable buffering in nginx
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Cache-Control')
    return response

@app.route('/api/action', methods=['POST', 'OPTIONS'])
def execute_action():
    """Execute an action in the active emulator"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("Action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        if not data:
            logger.error("No JSON data received in action request")
            return jsonify({"error": "No request data provided"}), 400

        action = data.get('action', 'SELECT')
        frames = data.get('frames', 1)

        # Validate action
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'}
        if action not in valid_actions:
            logger.warning(f"Invalid action requested: {action}")
            return jsonify({"error": f"Invalid action: {action}"}), 400

        logger.info(f"Executing action: {action} for {frames} frame(s)")
        emulator = emulators[game_state["active_emulator"]]
        success = emulator.step(action, frames)

        if success:
            action_history.append(action)
            logger.debug(f"Action {action} executed successfully")
            return jsonify({"message": "Action executed successfully"}), 200
        else:
            logger.error(f"Failed to execute action: {action}")
            return jsonify({"error": "Failed to execute action"}), 500

    except Exception as e:
        logger.error(f"Error executing action: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def get_ai_connector(api_name, api_key, api_endpoint):
    """Get an AI connector using the provider manager."""
    # Note: We now use the global ai_provider_manager for provider management
    # This function is kept for backward compatibility
    try:
        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key

        # Get the connector from the provider manager
        connector = ai_provider_manager.get_provider(api_name)
        if connector:
            logger.debug(f"Successfully obtained connector for {api_name}")
        else:
            logger.warning(f"Failed to obtain connector for {api_name}")
        return connector
    except Exception as e:
        logger.error(f"Error getting AI connector for {api_name}: {e}")
        return None

@app.route('/api/ai-action', methods=['POST', 'OPTIONS'])
def get_ai_action():
    """Get the next action from the specified AI API with automatic fallback"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("AI action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        if not data:
            logger.error("No JSON data received in AI action request")
            return jsonify({"error": "No request data provided"}), 400

        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')
        goal = data.get('goal', '')

        logger.info(f"AI action request: api={api_name or 'auto'}, model={model or 'default'}, goal='{goal}'")
        game_state["current_goal"] = goal

        emulator = emulators[game_state["active_emulator"]]
        screen_array = emulator.get_screen()

        if screen_array is None:
            logger.error("Failed to get screen from emulator")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert screen to bytes
        img_buffer = io.BytesIO()
        Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()

        if len(img_bytes) == 0:
            logger.error("Failed to convert screen to bytes")
            return jsonify({"error": "Failed to process screen image"}), 500

        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key
        if model:
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Use provider manager with automatic fallback
        logger.debug(f"Calling AI API: {api_name or 'auto'}")
        action, actual_provider = ai_provider_manager.get_next_action(
            img_bytes, goal, action_history, api_name, model
        )

        action_history.append(action)

        logger.info(f"AI ({actual_provider or 'fallback'}) suggested action: {action}")
        return jsonify({
            "action": action,
            "provider_used": actual_provider,
            "history": action_history[-10:]
        }), 200

    except Exception as e:
        logger.error(f"Error getting AI action: {e}", exc_info=True)
        api_name = data.get('api_name') if 'data' in locals() else 'unknown'
        goal = data.get('goal', '') if 'data' in locals() else 'unknown'
        logger.error(f"AI action failed - API: {api_name}, Goal: '{goal}', Screen size: {len(img_bytes) if 'img_bytes' in locals() else 'unknown'}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """Send a message to the AI and get a response with automatic fallback"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        user_message = data.get('message', '')
        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        emulator = emulators[game_state["active_emulator"]]
        screen_array = emulator.get_screen()

        img_buffer = io.BytesIO()
        Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()

        context = {
            "current_goal": game_state["current_goal"],
            "action_history": action_history[-20:],
            "game_type": game_state["active_emulator"].upper()
        }

        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key
        if model:
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Use provider manager with automatic fallback
        response_text, actual_provider = ai_provider_manager.chat_with_ai(
            user_message, img_bytes, context, api_name, model
        )

        logger.info(f"AI chat message from user: {user_message} (provider: {actual_provider or 'fallback'})")
        return jsonify({
            "response": response_text,
            "provider_used": actual_provider
        }), 200

    except Exception as e:
        logger.error(f"Error in AI chat: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/launch', methods=['POST'])
def launch_ui():
    """Launch UI process for the current ROM"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'launch_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.launch_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process launched successfully")
            return jsonify({
                "message": "UI launched successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to launch UI process")
            return jsonify({"error": "Failed to launch UI"}), 500

    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/stop', methods=['POST'])
def stop_ui():
    """Stop the UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'stop_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.stop_ui()

        if success:
            logger.info("UI process stopped successfully")
            return jsonify({"message": "UI stopped successfully"}), 200
        else:
            logger.error("Failed to stop UI process")
            return jsonify({"error": "Failed to stop UI"}), 500

    except Exception as e:
        logger.error(f"Error stopping UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/restart', methods=['POST'])
def restart_ui():
    """Restart the UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'restart_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.restart_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process restarted successfully")
            return jsonify({
                "message": "UI restarted successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to restart UI process")
            return jsonify({"error": "Failed to restart UI"}), 500

    except Exception as e:
        logger.error(f"Error restarting UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/status', methods=['GET'])
def get_ui_status():
    """Get UI process status"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'get_ui_status'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        ui_status = emulator.get_ui_status()

        return jsonify({
            "ui_status": ui_status,
            "rom_loaded": game_state["rom_loaded"],
            "active_emulator": game_state["active_emulator"]
        }), 200

    except Exception as e:
        logger.error(f"Error getting UI status: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/sync', methods=['POST'])
def sync_ui():
    """Synchronize server state with UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'sync_with_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.sync_with_ui()

        if success:
            logger.info("Server-UI synchronization completed")
            return jsonify({"message": "Synchronization completed"}), 200
        else:
            logger.warning("Server-UI synchronization failed or not supported")
            return jsonify({"error": "Synchronization failed"}), 500

    except Exception as e:
        logger.error(f"Error syncing with UI: {e}")
        return jsonify({"error": "Internal server error"}), 500


def main():
    """Main entry point for the server"""
    logger.info("Starting AI Game Server...")
    logger.info(f"Available AI providers: {ai_provider_manager.get_available_providers()}")

    if not ai_provider_manager.get_available_providers():
        logger.warning("[WARNING] No AI providers are available. AI features will be limited.")
        logger.info("To enable AI features, set the appropriate environment variables:")
        logger.info("  - GEMINI_API_KEY")
        logger.info("  - OPENROUTER_API_KEY")
        logger.info("  - NVIDIA_API_KEY (optional: NVIDIA_MODEL)")
        logger.info("  - OPENAI_API_KEY (optional: OPENAI_ENDPOINT for local providers)")
    else:
        logger.info(f"[SUCCESS] {len(ai_provider_manager.get_available_providers())} AI provider(s) are ready for use:")
        for provider_name in ai_provider_manager.get_available_providers():
            logger.info(f"  - {provider_name}")

    app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ == '__main__':
    main()
