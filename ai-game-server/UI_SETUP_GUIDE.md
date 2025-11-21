# PyBoy UI Integration Setup Guide

This guide provides comprehensive instructions for setting up and using the PyBoy UI integration system.

## Overview

The PyBoy UI integration system automatically launches a local PyBoy UI window whenever a ROM is loaded through the server, while maintaining the server's headless functionality for AI/remote control.

## Key Features

- **Automatic UI Launch**: UI window opens automatically when ROM is loaded
- **Dual Process Architecture**: Separate server and UI processes for stability
- **Sound Configuration**: Proper sound settings to prevent buffer overruns
- **REST API Control**: Full API control over UI processes
- **Configurable Settings**: Extensive configuration options
- **Robust Error Handling**: Graceful error recovery and process management

## System Requirements

- Python 3.8+
- PyBoy emulator (`pip install pyboy`)
- SDL2 libraries (`pip install pysdl2 pysdl2-dll`)
- Required server dependencies

## Installation

1. **Install PyBoy:**
```bash
pip install pyboy
```

2. **Install SDL2 dependencies:**
```bash
pip install pysdl2 pysdl2-dll
```

3. **Install server dependencies:**
```bash
cd ai-game-server
pip install -r requirements.txt
```

## Configuration

### Configuration File

The system creates a `ui_config.json` file with default settings. You can customize:

```json
{
  "window": {
    "type": "sdl2",
    "scale": 2,
    "color_palette": "grayscale",
    "fullscreen": false
  },
  "sound": {
    "enabled": true,
    "volume": 50,
    "sample_rate": 44100,
    "buffer_size": 1024,
    "emulated": true
  },
  "process": {
    "startup_timeout": 10,
    "shutdown_timeout": 5,
    "auto_restart": true,
    "cleanup_on_exit": true
  },
  "debug": {
    "log_level": "INFO",
    "show_fps": false,
    "enable_debug_overlay": false
  }
}
```

### Environment Variables

You can override configuration using environment variables:

```bash
export PYBOY_UI_SCALE=3
export PYBOY_UI_SOUND_VOLUME=75
export PYBOY_UI_WINDOW_TYPE=sdl2
export PYBOY_UI_LOG_LEVEL=DEBUG
```

## Usage

### Starting the Server

```bash
cd ai-game-server
python src/backend/server.py
```

### Loading a ROM with UI

1. **Via API (Recommended):**
```bash
curl -X POST \
  http://localhost:5000/api/upload-rom \
  -F "rom_file=@/path/to/rom.gb" \
  -F "emulator_type=gb" \
  -F "launch_ui=true"
```

2. **Via Web Interface:**
   - Navigate to your web interface
   - Upload a ROM file
   - UI will launch automatically

### UI Control API Endpoints

#### Launch UI
```bash
curl -X POST http://localhost:5000/api/ui/launch
```

#### Stop UI
```bash
curl -X POST http://localhost:5000/api/ui/stop
```

#### Restart UI
```bash
curl -X POST http://localhost:5000/api/ui/restart
```

#### Get UI Status
```bash
curl http://localhost:5000/api/ui/status
```

#### Synchronize Server-UI State
```bash
curl -X POST http://localhost:5000/api/ui/sync
```

## API Response Examples

### Successful ROM Upload with UI
```json
{
  "message": "ROM loaded successfully",
  "rom_name": "game.gb",
  "ui_launched": true,
  "ui_status": {
    "running": true,
    "ready": true,
    "rom_path": "/tmp/rom.gb",
    "pid": 12345
  }
}
```

### UI Status
```json
{
  "ui_status": {
    "running": true,
    "ready": true,
    "rom_path": "/tmp/rom.gb",
    "pid": 12345
  },
  "rom_loaded": true,
  "active_emulator": "gb"
}
```

## Troubleshooting

### UI Process Won't Start

1. **Check PyBoy Installation:**
```bash
python -c "import pyboy; print('PyBoy available')"
```

2. **Check SDL2 Installation:**
```bash
python -c "import sdl2; print('SDL2 available')"
```

3. **Check System Dependencies:**
   - Linux: `sudo apt-get install libsdl2-2.0-0`
   - macOS: SDL2 should be included with PyBoy
   - Windows: DLLs should be included with pysdl2-dll

### Sound Buffer Overruns

The system is configured to prevent sound buffer overruns:

- **Server**: Sound completely disabled (`sound=False`)
- **UI Process**: Properly configured sound settings
- **Configuration**: 44100 Hz sample rate, appropriate buffer size

If you still experience issues:

1. Check UI configuration:
```json
{
  "sound": {
    "enabled": true,
    "volume": 50,
    "sample_rate": 44100,
    "buffer_size": 1024
  }
}
```

2. Reduce volume if needed:
```bash
export PYBOY_UI_SOUND_VOLUME=30
```

### UI Process Management

The system includes robust process management:

- **Automatic Cleanup**: UI processes are cleaned up on exit
- **Graceful Shutdown**: Processes receive SIGTERM before SIGKILL
- **Restart Capability**: UI can be restarted without reloading ROM
- **Status Monitoring**: Real-time status tracking

### Logging

Enable debug logging for troubleshooting:

```bash
export PYBOY_UI_LOG_LEVEL=DEBUG
```

Or in configuration:
```json
{
  "debug": {
    "log_level": "DEBUG"
  }
}
```

## Advanced Usage

### Custom UI Launcher

You can create a custom UI launcher by extending the `ui_launcher.py` script:

```python
#!/usr/bin/env python3
import sys
from ui_launcher import parse_arguments, main

if __name__ == "__main__":
    # Add custom initialization here
    main()
```

### Multiple UI Instances

The system supports multiple UI instances for different ROMs:

```python
# Each emulator instance has its own UI manager
emulator1 = PyBoyEmulator()
emulator1.load_rom("rom1.gb")  # UI auto-launches

emulator2 = PyBoyEmulator()
emulator2.load_rom("rom2.gb")  # Separate UI instance
```

### Configuration Profiles

Create different configuration profiles for different use cases:

```python
# Development profile
dev_config = UIConfig("ui_config_dev.json")

# Production profile
prod_config = UIConfig("ui_config_prod.json")
```

## Testing

Run the integration test suite:

```bash
cd ai-game-server/src/backend
python test_ui_integration.py
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   AI Client     │    │   Mobile App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Flask Server         │
                    │  (Headless PyBoy Process)  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   UI Process Manager      │
                    │  (Process Control APIs)   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   PyBoy UI Process       │
                    │   (SDL2 Window + Sound)   │
                    └───────────────────────────┘
```

## Performance Considerations

- **Server**: Runs headless for maximum performance
- **UI Process**: Separate process with proper sound configuration
- **Memory**: Each process has its own memory space
- **CPU**: Dual processes utilize multiple cores efficiently

## Security Considerations

- **Process Isolation**: Server and UI run in separate processes
- **Input Validation**: All API inputs are validated
- **File Access**: Limited to ROM files and temporary directories
- **Network Access**: UI process doesn't require network access

## Contributing

To contribute to the UI integration system:

1. Run tests: `python test_ui_integration.py`
2. Follow existing code style
3. Add tests for new features
4. Update documentation

## Support

For issues and questions:

1. Check the troubleshooting section
2. Enable debug logging
3. Review the integration tests
4. Check system dependencies

---

This system provides a robust, configurable, and user-friendly way to automatically launch PyBoy UI windows while maintaining server stability and AI/remote control capabilities.