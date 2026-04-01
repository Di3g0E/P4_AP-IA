"""Configuracion centralizada del proyecto."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT.parent / "P2_AP-IA" / "data" / "raw" / "db_mod_descript.csv"

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Parametros del coach
MAX_HISTORY_TURNS = 10
MAX_RELEVANT_TRANSACTIONS = 20
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024
