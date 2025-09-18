# AI Game System - Complete Deployment Package

![AI Game System](https://img.shields.io/badge/AI-Game%20System-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Node.js](https://img.shields.io/badge/Node.js-18+-yellow) ![Windows](https://img.shields.io/badge/OS-Windows-blue)

A comprehensive multi-component emulation and AI gaming system featuring Game Boy and Game Boy Advance emulators with integrated AI APIs.

## 🚀 Quick Start

**Windows users:** Simply run `unified_startup.bat` for an automated setup experience!

```bash
# Double-click this file to start the system
unified_startup.bat
```

## 📋 System Requirements

### Minimum Requirements
- **Operating System:** Windows 10 or later
- **Python:** 3.8 or higher
- **Node.js:** 18.0 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 500MB free space

### Recommended Requirements
- **Operating System:** Windows 11
- **Python:** 3.10 or higher
- **Node.js:** 20.0 or higher
- **RAM:** 16GB or more
- **Storage:** 1GB free space
- **Internet Connection:** For AI API access

## 🛠️ Installation Instructions

### Method 1: Automated Installation (Recommended)

1. **Download the deployment package**
2. **Extract to your desired location**
3. **Run the unified startup script:**
   ```bash
   unified_startup.bat
   ```
4. **Follow the on-screen prompts**

### Method 2: Manual Installation

1. **Install Python:**
   - Download from [python.org](https://python.org)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install Node.js:**
   - Download from [nodejs.org](https://nodejs.org)
   - During installation, check "Add to PATH"
   - Verify installation: `node --version` and `npm --version`

3. **Install Dependencies:**
   ```bash
   # Install Python dependencies
   pip install -r ai-game-server/requirements.txt

   # Install Node.js dependencies
   cd ai-game-assistant
   npm install
   cd ..
   ```

## 🎯 Quick Start Guide

### First Time Setup

1. **Run the startup script:**
   ```bash
   unified_startup.bat
   ```

2. **Choose your startup mode:**
   - **Option 1:** Basic Startup (Quick start)
   - **Option 9:** Ultimate Mode (All features enabled)

3. **Wait for services to start:**
   - Backend server: http://localhost:5000
   - Frontend application: http://localhost:5173
   - Service Monitor: http://localhost:8080 (Ultimate mode only)

### Using the Web Interface

1. **Open your browser** and navigate to: http://localhost:5173
2. **Configure AI APIs** in the settings:
   - Add your API keys for Gemini, OpenRouter, or NVIDIA NIM
   - Select your preferred AI provider
3. **Load ROM files** or use built-in games
4. **Start playing** with AI assistance!

## 🔧 API Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# AI API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
NVIDIA_NIM_API_KEY=your_nvidia_nim_api_key_here

# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=true

# Port Configuration
BACKEND_PORT=5000
FRONTEND_PORT=5173
MONITOR_PORT=8080
```

### AI API Setup

#### Google Gemini
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your environment variables

#### OpenRouter
1. Go to [OpenRouter.ai](https://openrouter.ai/keys)
2. Create an account and generate an API key
3. Add it to your environment variables

#### NVIDIA NIM
1. Go to [NVIDIA NIM](https://build.nvidia.com/explore/reasoning)
2. Create an account and generate an API key
3. Add it to your environment variables

## 🚨 Troubleshooting Common Issues

### Python/Node.js Not Found
**Error:** `Python is not installed or not in PATH`
**Solution:**
1. Reinstall Python with "Add to PATH" option
2. Restart your computer after installation
3. Verify with: `python --version`

### Port Already in Use
**Error:** `Port 5000 is already in use`
**Solution:**
1. Close other applications using the port
2. Use different ports in configuration
3. Run "Clean Start" option in startup script

### API Key Issues
**Error:** `Invalid API key` or `Rate limit exceeded`
**Solution:**
1. Verify your API keys are correct
2. Check your API usage and billing
3. Try a different AI provider

### Emulator ROM Issues
**Error:** `ROM file not found` or `Invalid ROM format`
**Solution:**
1. Place ROM files in the correct directory
2. Ensure ROM files are in .gb/.gba format
3. Check ROM file integrity

### Frontend Not Loading
**Error:** Frontend shows blank page or errors
**Solution:**
1. Check Node.js installation
2. Clear browser cache
3. Try a different browser
4. Run "Development Mode" for detailed error logs

## 🎮 What's Included

### Core Components
- **PyBoy:** Game Boy emulator with Python API
- **PyGBA:** Game Boy Advance emulator wrapper
- **AI Game Server:** Backend API and AI integration
- **AI Game Assistant:** React-based web interface

### Features
- **Multi-Platform Emulation:** Play Game Boy and Game Boy Advance games
- **AI Integration:** Multiple AI providers for game assistance
- **Real-time Monitoring:** Service health and performance monitoring
- **Web Interface:** Modern, responsive UI for game control
- **Comprehensive Logging:** Detailed logging and debugging tools
- **Auto-Restart:** Automatic service recovery

### Startup Modes
- **Basic Startup:** Quick start with minimal features
- **Enhanced Logging:** Comprehensive logging and monitoring
- **Monitoring Console:** Real-time service monitoring
- **Service Monitor:** Auto-restart capabilities with web dashboard
- **Development Mode:** Verbose logging and debugging
- **Ultimate Mode:** All features enabled (recommended)

## 🔍 Advanced Features

### Service Monitoring
- Real-time service health checks
- Automatic service restart on failure
- Performance metrics and logging
- Web dashboard at http://localhost:8080

### Game Wrappers
Pre-built AI interfaces for popular games:
- Pokémon series
- Mario games
- Tetris
- And more!

### Development Tools
- Hot-reload for frontend development
- Verbose logging for debugging
- Comprehensive test suites
- API documentation

## 📚 Documentation

- [Deployment Guide](README_DEPLOYMENT.md) - Detailed deployment instructions
- [Startup Guide](STARTUP_GUIDE.md) - Startup script usage
- [Monitoring Guide](SERVICE_MONITORING_GUIDE.md) - Service monitoring setup
- [API Documentation](docs/API.md) - Backend API reference

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyBoy** - Game Boy emulator core
- **mGBA** - Game Boy Advance emulator
- **React & Vite** - Frontend framework
- **Flask** - Backend API framework
- **AI Providers** - Gemini, OpenRouter, NVIDIA NIM

---

**Need Help?**
- Check the troubleshooting section above
- Review the startup script output for error messages
- Verify your API keys and network connection
- Try different startup modes for debugging

**System Status:** ✅ All components operational
**Last Updated:** September 2024
**Version:** 2.0.0