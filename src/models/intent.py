"""Clasificador de intenciones por keywords para el coach financiero."""

import unicodedata
import re


def _normalize(text: str) -> str:
    """Normaliza texto: minusculas, sin acentos, sin caracteres especiales."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text


# Patrones de keywords por intent (ya normalizados)
_INTENT_PATTERNS: dict[str, list[str]] = {
    "savings_advice": [
        "ahorrar", "ahorro", "recortar", "gastar menos", "consejo",
        "reducir gastos", "como puedo gastar", "tips", "mejorar finanzas",
        "optimizar", "presupuesto",
    ],
    "concept_explanation": [
        "que es", "que son", "explica", "como funciona", "significado",
        "definicion", "diferencia entre", "que significa", "en que consiste",
        "interes compuesto", "diversificacion", "inflacion", "fondo de inversion",
        "letra del tesoro", "accion", "etf", "amortizacion",
    ],
    "spending_query": [
        "cuanto gasto", "cuanto he gastado", "resumen", "mes pasado",
        "tendencia", "total de gastos", "desglose", "categoria",
        "este mes", "ultimo mes", "ultimos meses", "cuanto llevo",
        "mis gastos", "mis ingresos", "cuanto ingreso", "balance",
    ],
}


def classify_intent(message: str) -> str:
    """Clasifica la intencion del mensaje del usuario.

    Returns:
        Uno de: 'savings_advice', 'concept_explanation', 'spending_query', 'general_chat'
    """
    normalized = _normalize(message)

    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in normalized)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general_chat"

    return max(scores, key=scores.get)
