"""Clase orquestadora del coach financiero generativo."""

from dataclasses import dataclass, field

import pandas as pd
from groq import Groq

from src.data.loader import load_transactions
from src.models.intent import classify_intent
from src.models.prompts import build_system_message
from src.utils.config import (
    DEFAULT_MODEL,
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
)


@dataclass
class CoachResponse:
    """Respuesta del coach financiero."""
    text: str
    intent: str
    model: str
    chart_data: pd.DataFrame | None = None
    chart_type: str | None = None
    transactions_table: pd.DataFrame | None = None


class FinancialCoach:
    """Coach financiero generativo que usa Groq API con modelos open-source."""

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = DEFAULT_MODEL,
        df: pd.DataFrame | None = None,
        data_path: str | None = None,
    ):
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "Configura tu GROQ_API_KEY en el archivo .env\n"
                "Obtener gratis en: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=api_key)
        self.model = model

        if df is not None:
            self.df = df
        elif data_path:
            self.df = load_transactions(data_path)
        else:
            from src.utils.config import DATA_PATH
            self.df = load_transactions(DATA_PATH)

    def chat(
        self,
        message: str,
        conversation_history: list[dict] | None = None,
    ) -> CoachResponse:
        """Procesa un mensaje del usuario y genera una respuesta personalizada.

        Args:
            message: Mensaje del usuario en espanol.
            conversation_history: Lista de mensajes previos [{"role": ..., "content": ...}].

        Returns:
            CoachResponse con texto, intent, y datos opcionales para graficos.
        """
        if conversation_history is None:
            conversation_history = []

        # 1. Clasificar intencion
        intent = classify_intent(message)

        # 2. Construir system message con contexto relevante
        system_msg = build_system_message(self.df, intent)

        # 3. Ensamblar mensajes para la API
        messages = [{"role": "system", "content": system_msg}]

        # Anadir historial (ventana deslizante)
        recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(recent_history)

        # Anadir mensaje actual
        messages.append({"role": "user", "content": message})

        # 4. Ajustar temperatura segun intent
        temperature = LLM_TEMPERATURE
        if intent == "spending_query":
            temperature = 0.3  # Mas preciso para datos numericos

        # 5. Llamar a la API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
        )

        response_text = response.choices[0].message.content

        # 6. Preparar datos para graficos si aplica
        chart_data = None
        chart_type = None
        if intent == "spending_query":
            from src.features.analytics import compute_category_breakdown
            breakdown = compute_category_breakdown(self.df)
            chart_data = pd.DataFrame([
                {"Categoria": cat, "Total EUR": info["total"], "% del Total": info["pct"]}
                for cat, info in breakdown["categories"].items()
            ])
            chart_type = "bar"

        return CoachResponse(
            text=response_text,
            intent=intent,
            model=self.model,
            chart_data=chart_data,
            chart_type=chart_type,
        )

    def reload_data(self, path: str | None = None) -> None:
        """Recarga los datos del CSV (para cuando se anaden nuevas transacciones)."""
        if path:
            self.df = load_transactions(path)
        else:
            from src.utils.config import DATA_PATH
            self.df = load_transactions(DATA_PATH)
