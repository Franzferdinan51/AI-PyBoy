import React, { useState, useRef, useEffect, useCallback } from 'react';
import Header from './components/Header';
import EmulatorScreen from './components/EmulatorScreen';
import Controls from './components/Controls';
import AIPanel from './components/AIPanel';
import SettingsModal from './components/SettingsModal';
import type { EmulatorMode, AIState, GameAction, AILog, ChatMessage, AppSettings } from './types';
import { EmulatorMode as EmulatorModeEnum, AIState as AIStateEnum } from './types';

const SERVER_URL = 'http://localhost:5000';

const App: React.FC = () => {
  const [emulatorMode, setEmulatorMode] = useState<EmulatorMode>(EmulatorModeEnum.GB);
  const [romName, setRomName] = useState<string | null>(null);
  const [aiState, setAiState] = useState<AIState>(AIStateEnum.IDLE);
  const [aiGoal, setAiGoal] = useState<string>('');
  const [aiLogs, setAiLogs] = useState<AILog[]>([]);
  const [lastAIAction, setLastAIAction] = useState<GameAction | null>(null);
  const [screenImage, setScreenImage] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatting, setIsChatting] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error' | 'failed'>('disconnected');
  const [streamingInfo, setStreamingInfo] = useState<{ fps: number; frameCount: number }>({ fps: 0, frameCount: 0 });
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [appSettings, setAppSettings] = useState<AppSettings>(() => {
    const savedSettings = localStorage.getItem('appSettings');
    if (savedSettings) {
      return JSON.parse(savedSettings);
    }
    return {
      aiActionInterval: 5000,
      apiProvider: 'gemini',
    };
  });

  const gameLoopRef = useRef<number | null>(null);
  const actionHistoryRef = useRef<string[]>([]);
  const logIdCounter = useRef<number>(0);
  const chatIdCounter = useRef<number>(0);

  const handleOpenSettings = () => setIsSettingsOpen(true);
  const handleCloseSettings = () => setIsSettingsOpen(false);

  const handleSaveSettings = (newSettings: AppSettings) => {
    setAppSettings(newSettings);
    localStorage.setItem('appSettings', JSON.stringify(newSettings));
    if (gameLoopRef.current) {
      stopAI();
    }
    handleCloseSettings();
  };

  useEffect(() => {
    let source: EventSource | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connectSSE = () => {
      if (!romName) return;

      try {
        source = new EventSource(`${SERVER_URL}/api/stream`);

        source.onopen = () => {
          console.log('SSE connection established');
          reconnectAttempts = 0;
        };

        source.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.error) {
              console.error('SSE stream error:', data.error);
              setStreamingStatus('error');
              return;
            }

            if (data.image) {
              setScreenImage(`data:image/jpeg;base64,${data.image}`);
              setStreamingStatus('connected');

              // Update FPS counter if available
              if (data.fps) {
                setStreamingInfo(prev => ({ ...prev, fps: data.fps }));
              }
            }

            if (data.status === 'stream_started') {
              console.log('SSE stream started successfully');
              setStreamingStatus('connected');
            }
          } catch (e) {
            console.error('SSE parse error:', e);
            setStreamingStatus('error');
          }
        };

        source.onerror = (err) => {
          console.error('SSE error:', err);
          setStreamingStatus('disconnected');
          source?.close();

          // Attempt to reconnect
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect... (${reconnectAttempts}/${maxReconnectAttempts})`);
            reconnectTimeout = setTimeout(connectSSE, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
          } else {
            console.error('Max reconnection attempts reached');
            setStreamingStatus('failed');
          }
        };

        setEventSource(source);
      } catch (error) {
        console.error('Failed to create SSE connection:', error);
        setStreamingStatus('failed');
      }
    };

    connectSSE();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      source?.close();
      setEventSource(null);
      setStreamingStatus('disconnected');
    };
  }, [romName]);

  const addLog = useCallback((message: string, type: AILog['type']) => {
    setAiLogs(prev => [...prev, { id: logIdCounter.current++, message, type }]);
  }, []);

  const addChatMessage = useCallback((message: string, sender: 'user' | 'ai') => {
    setChatHistory(prev => [...prev, { id: chatIdCounter.current++, text: message, sender }]);
  }, []);

  // Function to update the screen from the server
  const updateScreen = useCallback(async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/screen`);
      if (!response.ok) {
        throw new Error('Failed to get screen');
      }
      
      const data = await response.json();
      setScreenImage(`data:image/jpeg;base64,${data.image}`);
    } catch (error) {
      console.error('Error updating screen:', error);
    }
  }, []);

  // Function to load a ROM
  const loadRom = useCallback(async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('rom_file', file);
      formData.append('emulator_type', emulatorMode);
      
      setLoading(true);
      addLog(`Uploading ROM: ${file.name}`, 'info');
      
      const response = await fetch(`${SERVER_URL}/api/upload-rom`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }
      
      const result = await response.json();
      setRomName(file.name);
      addLog(`Loaded ROM: ${file.name}`, 'info');
      console.log('ROM loaded:', result);
      
      await updateScreen();
      setLoading(false);
    } catch (error) {
      console.error('Error loading ROM:', error);
      addLog(`Error loading ROM: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      setLoading(false);
    }
  }, [emulatorMode, addLog, updateScreen]);

  // Function to execute an action
  const executeAction = useCallback(async (action: string) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: action,
          frames: 10
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to execute action');
      }
      
      await updateScreen();
    } catch (error) {
      console.error('Error executing action:', error);
      addLog(`Error executing action: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog, updateScreen]);

  // Function to get AI next move
  const getAINextMove = useCallback(async (goal: string) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/ai-action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_name: appSettings.apiProvider,
          api_endpoint: appSettings.apiEndpoint,
          api_key: appSettings.apiKey,
          model: appSettings.model,
          goal: goal
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get AI action');
      }
      
      const data = await response.json();
      return data.action;
    } catch (error) {
      console.error('Error getting AI action:', error);
      addLog(`Error getting AI action: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      return 'SELECT'; // Default safe action
    }
  }, [addLog, appSettings]);

  const runAI = useCallback(async () => {
    if (!romName) {
      addLog("Cannot start AI: No ROM loaded.", 'error');
      setAiState(AIStateEnum.IDLE);
      return;
    }
    setAiState(AIStateEnum.THINKING);
    addLog("AI is thinking...", 'thought');

    try {
      const nextAction = await getAINextMove(aiGoal);
      
      actionHistoryRef.current.push(nextAction);
      addLog(`AI chose action: ${nextAction}`, 'action');
      setLastAIAction(nextAction as GameAction);
      
      await executeAction(nextAction);
      
      setTimeout(() => setLastAIAction(null), 500);
      setAiState(AIStateEnum.RUNNING);

    } catch (error) {
      console.error(error);
      const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
      addLog(`Error: ${errorMessage}`, 'error');
      setAiState(AIStateEnum.ERROR);
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
        gameLoopRef.current = null;
      }
    }
  }, [romName, aiGoal, addLog, getAINextMove, executeAction]);

  const startAI = () => {
    if (gameLoopRef.current) return;
    setAiLogs([]);
    actionHistoryRef.current = [];
    addLog(`AI started with objective: "${aiGoal}"`, 'info');
    setAiState(AIStateEnum.RUNNING);
    runAI();
    gameLoopRef.current = window.setInterval(runAI, appSettings.aiActionInterval);
  };

  const stopAI = () => {
    if (gameLoopRef.current) {
      clearInterval(gameLoopRef.current);
      gameLoopRef.current = null;
    }
    addLog("AI stopped by user.", 'info');
    setAiState(AIStateEnum.IDLE);
  };
  
  const handleRomLoad = (file: File) => {
    setRomName(file.name);
    addLog(`Loaded ROM: ${file.name}`, 'info');
    loadRom(file);
  };

  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    addChatMessage(message, 'user');
    setChatInput('');
    setIsChatting(true);

    try {
      const response = await fetch(`${SERVER_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          api_name: appSettings.apiProvider,
          api_endpoint: appSettings.apiEndpoint,
          api_key: appSettings.apiKey,
          model: appSettings.model,
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get chat response');
      }

      const data = await response.json();
      addChatMessage(data.response, 'ai');
    } catch (error) {
      console.error('Error sending chat message:', error);
      addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
    } finally {
      setIsChatting(false);
    }
  }, [addChatMessage, appSettings]);

  useEffect(() => {
    // Add event listener for manual control input
    const handleControlPress = (event: CustomEvent) => {
      const action = event.detail;
      if (romName) {
        executeAction(action);
        // Briefly show the action as active for visual feedback
        setLastAIAction(action as GameAction);
        setTimeout(() => setLastAIAction(null), 200);
      }
    };

    window.addEventListener('game-control-press', handleControlPress as EventListener);

    return () => {
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
      }
      window.removeEventListener('game-control-press', handleControlPress as EventListener);
    };
  }, [romName, executeAction]);

  return (
    <div className="h-screen w-screen flex flex-col bg-neutral-950">
      <Header emulatorMode={emulatorMode} onModeChange={setEmulatorMode} onOpenSettings={handleOpenSettings} />
      <main className="flex-grow flex flex-col md:flex-row min-h-0">
        <div className="flex-grow flex flex-col items-center bg-neutral-950 md:p-4">
          <div className="w-full h-full max-w-7xl flex flex-col bg-neutral-900 rounded-lg shadow-2xl shadow-black/50">
            <EmulatorScreen
              emulatorMode={emulatorMode}
              romName={romName}
              onRomLoad={handleRomLoad}
              aiState={aiState}
              screenImage={screenImage}
              streamingStatus={streamingStatus}
              streamingInfo={streamingInfo}
            />
            <Controls lastAction={lastAIAction} />
          </div>
        </div>
        <AIPanel
          aiState={aiState}
          aiLogs={aiLogs}
          aiGoal={aiGoal}
          chatHistory={chatHistory}
          chatInput={chatInput}
          isChatting={isChatting}
          onGoalChange={setAiGoal}
          onStart={startAI}
          onStop={stopAI}
          onChatInputChange={setChatInput}
          onSendMessage={() => handleSendMessage(chatInput)}
        />
      </main>
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={handleCloseSettings}
        currentSettings={appSettings}
        onSave={handleSaveSettings}
      />
    </div>
  );
};

export default App;
