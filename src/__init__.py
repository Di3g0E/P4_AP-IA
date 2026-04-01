"""Coach Financiero Generativo - Modulos principales."""

from src.data.loader import load_transactions
from src.data.retrieval import (
    format_transactions_for_prompt,
    get_recent_transactions,
    get_top_expenses,
    get_transactions_by_area,
    get_transactions_by_date_range,
    search_transactions,
)
from src.features.analytics import (
    compute_category_breakdown,
    compute_investment_summary,
    compute_monthly_summary,
    compute_recurring_expenses,
    compute_savings_rate,
    compute_spending_trends,
    detect_anomalies,
)
from src.models.coach import FinancialCoach
from src.models.intent import classify_intent
