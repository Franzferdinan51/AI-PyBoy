"""
Abstract interface for game emulators
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import numpy as np


class EmulatorInterface(ABC):
    """Abstract base class for game emulators"""
    
    @abstractmethod
    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the emulator"""
        pass
    
    @abstractmethod
    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames"""
        pass
    
    @abstractmethod
    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array"""
        pass
    
    @abstractmethod
    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        pass
    
    @abstractmethod
    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the emulator"""
        pass
    
    @abstractmethod
    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        pass
    
    @abstractmethod
    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Get information about the current game state"""
        pass
    
    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        # Default implementation
        return {
            "basic_info": self.get_info(),
            "screen_analysis": {},
            "memory_analysis": {},
            "game_specific": {}
        }