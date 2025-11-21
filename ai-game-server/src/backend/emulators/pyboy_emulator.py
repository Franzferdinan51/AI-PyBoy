"""
PyBoy emulator implementation
"""
import numpy as np
from typing import List, Tuple, Optional, Any
import io
import os
import time

# Try to import PyBoy
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Install with 'pip install pyboy'")

from .emulator_interface import EmulatorInterface
try:
    from ..ui_process_manager import get_ui_manager
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ui_process_manager import get_ui_manager


class PyBoyEmulator(EmulatorInterface):
    """PyBoy emulator implementation"""
    
    def __init__(self):
        self.pyboy = None
        self.rom_path = None
        self.initialized = False
        self.game_title = ""
        self.ui_manager = get_ui_manager()
        
    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the PyBoy emulator"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy is not available. Please install it with 'pip install pyboy'")

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        try:
            # Initialize PyBoy in headless mode for server stability
            self.pyboy = PyBoy(
                rom_path,
                window="null",  # Headless mode for server
                sound=False,    # Disable sound in server
                log_level="WARNING",
                sound_emulated=False,  # Don't emulate sound registers
                sound_volume=0       # Zero volume
            )
            self.rom_path = rom_path
            self.initialized = True
            self.game_title = self.pyboy.cartridge_title if self.pyboy else ""

            # Launch UI automatically using PyBoy's native SDL2 support
            self._launch_ui_native(rom_path)

            return True
        except Exception as e:
            print(f"Error loading ROM: {e}")
            return False
    
    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames"""
        if not self.initialized or self.pyboy is None:
            return False

        # Map actions to PyBoy buttons
        action_map = {
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right',
            'A': 'a',
            'B': 'b',
            'START': 'start',
            'SELECT': 'select'
        }

        try:
            # Process each frame
            for frame in range(frames):
                if action in action_map:
                    # Press the button, then tick to process the input
                    self.pyboy.button(action_map[action])
                    # Process one frame with button pressed
                    self.pyboy.tick(1, True)
                else:
                    # For NOOP actions, just tick the emulator
                    self.pyboy.tick(1, True)

                # Release button after processing (if it was a button action)
                if action in action_map and frame == frames - 1:
                    # Release the button
                    button_actions = {
                        'up': 'UP',
                        'down': 'DOWN',
                        'left': 'LEFT',
                        'right': 'RIGHT',
                        'a': 'A',
                        'b': 'B',
                        'start': 'START',
                        'select': 'SELECT'
                    }
                    if action_map[action] in button_actions:
                        release_action = button_actions[action_map[action]]
                        self.pyboy.button_release(action_map[action])
                        self.pyboy.tick(1, True)

            return True
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False
    
    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array"""
        if not self.initialized or self.pyboy is None:
            return np.zeros((144, 160, 3), dtype=np.uint8)

        try:
            # Get screen as numpy array directly
            screen_array = self.pyboy.screen.ndarray

            # Validate screen array dimensions
            if screen_array is None or screen_array.size == 0:
                return np.zeros((144, 160, 3), dtype=np.uint8)

            # Convert from RGBA to RGB if needed
            if screen_array.shape[2] == 4:
                # Remove alpha channel
                screen_array = screen_array[:, :, :3]

            # Ensure RGB format
            if screen_array.shape[2] != 3:
                # Convert grayscale to RGB if needed
                if len(screen_array.shape) == 2:
                    screen_array = np.stack([screen_array] * 3, axis=2)
                else:
                    return np.zeros((144, 160, 3), dtype=np.uint8)

            return screen_array.astype(np.uint8)
        except Exception as e:
            print(f"Error getting screen: {e}")
            return np.zeros((144, 160, 3), dtype=np.uint8)
    
    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        if not self.initialized or self.pyboy is None:
            return b'\x00' * size
            
        try:
            if size == 1:
                return bytes([self.pyboy.memory[address]])
            else:
                return bytes(self.pyboy.memory[address:address + size])
        except Exception as e:
            print(f"Error reading memory at {hex(address)}: {e}")
            return b'\x00' * size
    
    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        if not self.initialized or self.pyboy is None:
            return False
            
        try:
            if len(value) == 1:
                self.pyboy.memory[address] = value[0]
            else:
                self.pyboy.memory[address:address + len(value)] = list(value)
            return True
        except Exception as e:
            print(f"Error writing memory at {hex(address)}: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset the emulator"""
        if not self.initialized or self.pyboy is None:
            return False
            
        try:
            # Stop the current emulator
            self.pyboy.stop()

            # Re-initialize PyBoy in headless mode
            self.pyboy = PyBoy(
                self.rom_path,
                window="null",
                sound=False,
                log_level="WARNING",
                # Additional stability settings
                sound_emulated=False,  # Don't emulate sound registers
                sound_volume=0       # Zero volume
            )
            self.game_title = self.pyboy.cartridge_title if self.pyboy else ""
            return True
        except Exception as e:
            print(f"Error resetting emulator: {e}")
            return False
    
    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        if not self.initialized or self.pyboy is None:
            return b''
            
        try:
            # Save state to bytes
            state_buffer = io.BytesIO()
            self.pyboy.save_state(state_buffer)
            return state_buffer.getvalue()
        except Exception as e:
            print(f"Error saving state: {e}")
            return b''
    
    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        if not self.initialized or self.pyboy is None:
            return False
            
        try:
            # Load state from bytes
            state_buffer = io.BytesIO(state)
            self.pyboy.load_state(state_buffer)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def get_info(self) -> dict:
        """Get information about the current game state"""
        if not self.initialized or self.pyboy is None:
            return {}
            
        try:
            return {
                "rom_title": self.pyboy.cartridge_title,
                "frame_count": self.pyboy.frame_count,
                "screen_size": self.pyboy.screen.ndarray.shape,
                "initialized": self.initialized,
                "game_title": self.game_title
            }
        except Exception as e:
            print(f"Error getting info: {e}")
            return {}
    
    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        if not self.initialized or self.pyboy is None:
            return {}
            
        try:
            # Get basic info
            info = self.get_info()
            
            # Get screen analysis
            screen = self.get_screen()
            
            # Get memory regions of interest (this would be game-specific)
            # For now, we'll just get some general memory values
            memory_analysis = {}
            
            # Add game-specific analysis based on the game title
            game_specific = self._get_game_specific_analysis()
            
            return {
                "basic_info": info,
                "screen_analysis": {
                    "shape": screen.shape,
                    "mean_color": screen.mean(axis=(0,1)).tolist(),
                    "unique_colors": len(np.unique(screen.reshape(-1, screen.shape[2]), axis=0))
                },
                "memory_analysis": memory_analysis,
                "game_specific": game_specific
            }
        except Exception as e:
            print(f"Error getting game state analysis: {e}")
            return {}
    
    def _get_game_specific_analysis(self) -> dict:
        """Get game-specific analysis based on the game title"""
        if not self.initialized or self.pyboy is None:
            return {}
            
        game_title = self.game_title.lower()
        
        # Placeholder for game-specific analysis
        # In a real implementation, this would have specific logic for each game
        if "pokemon" in game_title:
            return self._get_pokemon_analysis()
        elif "tetris" in game_title:
            return self._get_tetris_analysis()
        elif "mario" in game_title:
            return self._get_mario_analysis()
        else:
            return {"game_type": "unknown", "analysis": "No specific analysis available for this game"}
    
    def _get_pokemon_analysis(self) -> dict:
        """Get Pokemon-specific game analysis"""
        # Placeholder for Pokemon-specific analysis
        # This would read memory addresses specific to Pokemon games
        return {
            "game_type": "pokemon",
            "player_position": "unknown",
            "current_party": [],
            "battle_status": "unknown"
        }
    
    def _get_tetris_analysis(self) -> dict:
        """Get Tetris-specific game analysis"""
        # Placeholder for Tetris-specific analysis
        return {
            "game_type": "tetris",
            "current_piece": "unknown",
            "next_piece": "unknown",
            "lines_cleared": 0
        }
    
    def _get_mario_analysis(self) -> dict:
        """Get Mario-specific game analysis"""
        # Placeholder for Mario-specific analysis
        return {
            "game_type": "mario",
            "player_position": "unknown",
            "lives": 0,
            "coins": 0
        }

    def _launch_ui_native(self, rom_path: str) -> bool:
        """Launch UI using PyBoy's native SDL2 support"""
        try:
            import subprocess
            import sys

            # Use PyBoy's native SDL2 window in a separate process
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, r'{os.path.dirname(os.path.dirname(__file__))}')
try:
    from pyboy import PyBoy
    import os

    # Configure UI with proper sound settings
    ui_pyboy = PyBoy(
        r'{rom_path}',
        window='SDL2',
        scale=2,
        sound=True,
        sound_volume=50,
        sound_emulated=True,
        log_level='INFO',
        color_palette='grayscale'
    )

    print('UI_READY')

    # Run the UI
    while ui_pyboy.tick():
        pass

    ui_pyboy.stop()

except Exception as e:
    print(f'UI Error: {{e}}')
    import traceback
    traceback.print_exc()
"""
            ]

            # Launch UI process
            self.ui_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            print(f"UI launched for ROM: {rom_path}")
            return True

        except Exception as e:
            print(f"Error launching UI: {e}")
            return False

    def launch_ui(self) -> bool:
        """Launch UI for current ROM"""
        if not self.initialized or not self.rom_path:
            print("No ROM loaded to launch UI for")
            return False

        return self._launch_ui_native(self.rom_path)

    def stop_ui(self) -> bool:
        """Stop the UI process"""
        try:
            if hasattr(self, 'ui_process') and self.ui_process:
                self.ui_process.terminate()
                try:
                    self.ui_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ui_process.kill()
                self.ui_process = None
                print("UI stopped")
            return True
        except Exception as e:
            print(f"Error stopping UI: {e}")
            return False

    def restart_ui(self) -> bool:
        """Restart the UI process"""
        self.stop_ui()
        if self.rom_path:
            return self._launch_ui_native(self.rom_path)
        return False

    def get_ui_status(self) -> dict:
        """Get UI process status"""
        if hasattr(self, 'ui_process') and self.ui_process:
            return {
                "running": self.ui_process.poll() is None,
                "pid": self.ui_process.pid
            }
        return {"running": False, "pid": None}

    def sync_with_ui(self) -> bool:
        """Synchronize server state with UI process"""
        # For native UI, this is handled automatically
        return True

    def __del__(self):
        """Cleanup when emulator is destroyed"""
        try:
            self.stop_ui()
        except:
            pass