#!/usr/bin/env python3
"""
Bass Tab Transcriber
Entry point.
"""
import sys
import os

# Make sure we can import from the src package
sys.path.insert(0, os.path.dirname(__file__))

from src.app import run

if __name__ == "__main__":
    run()
