#!/usr/bin/env python3
"""Lazy, CPU-safe Seek entry point. Real model/environment imports are phase-local."""
from pathlib import Path
import sys

WEBSHOP = Path(__file__).resolve().parent / "agent-backdoor-attacks/AgentTuning/WebShop"
sys.path.insert(0, str(WEBSHOP))

from seek.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
