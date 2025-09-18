# AI Game System - Deployment Guide

## 🚀 Deployment Instructions

### Step 1: System Preparation
1. **Extract the deployment package** to your desired location
2. **Ensure Windows 10/11** is installed and updated
3. **Verify administrative privileges** for installation

### Step 2: Software Installation
#### Method A: Automated (Recommended)
```bash
# Run the automated dependency installer
install_dependencies.bat
```

#### Method B: Manual Installation
1. **Install Python 3.8+**
   - Download from [python.org](https://python.org)
   - Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Install Node.js 18+**
   - Download from [nodejs.org](https://nodejs.org)
   - Check "Add to PATH" during installation
   - Verify: `node --version` and `npm --version`

### Step 3: Environment Configuration
Create a `.env` file in the `ai-game-server` directory based on `.env.example`:
```env
# AI API Keys (Optional - system works without them)
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
NVIDIA_API_KEY=your_nvidia_key_here
OPENAI_API_KEY=your_openai_key_here
OPENAI_ENDPOINT=your_openai_endpoint_here

# Optional Settings
NVIDIA_MODEL=nvidia/llama3-llm-70b
```

### Step 4: First-Time Startup
1. **Run the startup script:**
   ```bash
   unified_startup.bat
   ```

2. **Choose startup mode:**
   - **Option 1**: Basic Startup (Quick start)
   - **Option 9**: Ultimate Mode (All features)

3. **Wait for services to start:**
   - Backend: http://localhost:5000
   - Frontend: http://localhost:5173
   - Monitor: http://localhost:8080 (Ultimate mode)

### Step 5: Verification
1. **Open browser** to http://localhost:5173
2. **Test backend** at http://localhost:5000/health
3. **Check service monitor** at http://localhost:8080 (if enabled)

## 🔧 Advanced Configuration

### Port Configuration
Default ports can be modified in the `.env` file:
```env
BACKEND_PORT=5000
FRONTEND_PORT=5173
MONITOR_PORT=8080
```

### AI API Configuration
The system works without API keys, but for AI features:

1. **Gemini API** (Free tier available)
   - Get key: https://makersuite.google.com/app/apikey
   - Set: `GEMINI_API_KEY=your_key`

2. **OpenRouter API** (Paid)
   - Get key: https://openrouter.ai/keys
   - Set: `OPENROUTER_API_KEY=your_key`

3. **NVIDIA NIM API** (Paid)
   - Get key: https://build.nvidia.com/
   - Set: `NVIDIA_API_KEY=your_key`

### Firewall Configuration
Ensure Windows Firewall allows:
- Port 5000 (Backend API)
- Port 5173 (Frontend)
- Port 8080 (Service Monitor)

## 🚨 Troubleshooting

### Common Issues
1. **Port conflicts**: Use different ports in `.env` file
2. **Missing dependencies**: Run `install_dependencies.bat` again
3. **API key issues**: Verify keys in `.env` file
4. **Permission errors**: Run as administrator

### Log Files
Check these locations for detailed logs:
- `ai-game-server/ai_game_server.log` for backend logs
- Browser console for frontend errors

## 🎮 Usage Instructions

### Loading Games
1. **Open web interface** at http://localhost:5173
2. **Upload ROM files** (.gb, .gbc, .gba)
3. **Configure AI settings** if desired
4. **Start playing!**

### AI Features
- **Game assistance**: AI can suggest moves and strategies
- **Chat interface**: Talk to AI about the game
- **Automated play**: AI can play games autonomously

### Manual Controls
If AI isn't working, you can still:
- Use manual controls through the web interface
- Execute actions directly via API calls
- Play games normally without AI assistance

## 📱 System Requirements

### Minimum
- Windows 10/11 (64-bit)
- 4GB RAM
- 500MB storage
- Python 3.8+
- Node.js 18+

### Recommended
- Windows 11
- 8GB+ RAM
- 1GB storage
- Python 3.10+
- Node.js 20+
- Internet connection for AI APIs

## 🔧 Maintenance

### Updates
1. **Check for updates** regularly
2. **Backup your `.env` file** before updating
3. **Run dependency installers** after updates

### Backup
- **Save ROM files** separately
- **Backup `.env` file** with API keys
- **Export game saves** if needed

## 📞 Support

### Getting Help
1. **Check this guide** first
2. **Review log files** for error details
3. **Try different startup modes** for debugging
4. **Verify system requirements**

### Common Solutions
- **Restart services** using the startup script
- **Clear browser cache** for frontend issues
- **Reinstall dependencies** for persistent errors
- **Check Windows Firewall** for connection issues

---

## 🎯 Key Features Working

### ✅ **Fixed Issues:**
- **ROM Streaming**: Now shows live gameplay instead of still images
- **AI Action Execution**: All actions (SELECT, START, A, B, UP, DOWN, LEFT, RIGHT, NOOP) work
- **API Endpoints**: All endpoints functioning with proper error handling
- **CORS Configuration**: Frontend-backend communication working
- **Error Handling**: Graceful degradation when API keys aren't available

### ✅ **Current Capabilities:**
- **Live Gameplay Streaming**: 30 FPS real-time gameplay
- **Manual Controls**: All game controls working through web interface
- **AI Integration**: Ready for API keys (returns default actions without keys)
- **ROM Support**: Game Boy, Game Boy Color, Game Boy Advance
- **Web Interface**: Modern, responsive design

### ✅ **What Works Without API Keys:**
- **Game streaming and display**
- **Manual game controls**
- **ROM loading and management**
- **Save/load states**
- **All basic emulator functionality**

### 🔑 **What Requires API Keys:**
- **AI-powered gameplay assistance**
- **Game strategy suggestions**
- **Automated AI play**
- **Game analysis and chat**

---

**Deployment Complete!** 🎉

Your AI Game System is now fully functional with all critical issues resolved. The system works immediately without API keys, but you can add them later for enhanced AI features.