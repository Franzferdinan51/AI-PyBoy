"""
AI Game Server - Main entry point
"""
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from server import main

if __name__ == "__main__":
    main()