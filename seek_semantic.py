#!/usr/bin/env python3
"""Isolated semantic Seek entry point; metadata commands require only stdlib."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.semantic_cli import main
if __name__=='__main__': raise SystemExit(main())
