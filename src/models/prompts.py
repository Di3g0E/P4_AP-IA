"""Templates de prompts y ensamblaje de contexto para el coach financiero."""

import pandas as pd

from src.data.retrieval import (
    format_transactions_for_prompt,
    get_recent_transactions,
    get_transactions_by_area,
    get_top_expenses,
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


SYSTEM_PROMPT_TEMPLATE = """\
Eres un coach financiero personal experto y amable. Siempre respondes en espanol.

REGLAS ESTRICTAS:
- Basa tus respuestas EXCLUSIVAMENTE en los datos financieros reales del usuario proporcionados abajo.
- NUNCA inventes transacciones, cifras o datos que no aparezcan en el contexto.
- Cuando expliques conceptos financieros, usa ejemplos concretos de las transacciones del usuario.
- Se motivador y practico. Da consejos especificos y accionables.
- Formatea cantidades en EUR con dos decimales.
- Si no tienes datos suficientes para responder, dilo honestamente.
- Responde de forma concisa pero completa.

PERFIL FINANCIERO DEL USUARIO:
{profile}

{context_block}"""


def build_user_profile(df: pd.DataFrame) -> str:
    """Construye un resumen compacto del perfil financiero del usuario."""
    expenses = df[df["Type"] == "Expenses"]
    income = df[df["Type"] == "Income"]
    total_inc = income["Amount_clean"].sum()
    total_exp = expenses["Amount_clean"].sum()

    # Breakdown por categoria
    cat_summary = (
        expenses.groupby("Area")["Amount_clean"]
        .sum()
        .sort_values(ascending=False)
    )
    cat_lines = [
        f"  - {area}: {amount:.2f} EUR ({amount / total_exp * 100:.1f}%)"
        for area, amount in cat_summary.items()
    ]

    # Ultimos 3 meses
    recent_periods = sorted(df["YearMonth"].unique(), reverse=True)[:3]
    recent = df[df["YearMonth"].isin(recent_periods)]
    recent_exp = recent[recent["Type"] == "Expenses"]["Amount_clean"].sum()
    recent_inc = recent[recent["Type"] == "Income"]["Amount_clean"].sum()

    return (
        f"Periodo de datos: {df['Date_parsed'].min().strftime('%d/%m/%Y')} - "
        f"{df['Date_parsed'].max().strftime('%d/%m/%Y')}\n"
        f"Total transacciones: {len(df)}\n"
        f"Ingresos totales: {total_inc:,.2f} EUR\n"
        f"Gastos totales: {total_exp:,.2f} EUR\n"
        f"Ahorro neto: {total_inc - total_exp:,.2f} EUR\n"
        f"Tasa de ahorro global: "
        f"{((total_inc - total_exp) / total_inc * 100) if total_inc > 0 else 0:.1f}%\n"
        f"\nDesglose de gastos por categoria:\n" + "\n".join(cat_lines) + "\n"
        f"\nUltimos 3 meses:\n"
        f"  Ingresos: {recent_inc:,.2f} EUR\n"
        f"  Gastos: {recent_exp:,.2f} EUR\n"
        f"  Ahorro: {recent_inc - recent_exp:,.2f} EUR"
    )


def _format_analytics_block(data: dict, title: str) -> str:
    """Formatea un dict de analytics como bloque de texto."""
    lines = [f"\n{title}:"]
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    - {k}: {v}")
        elif isinstance(value, list):
            lines.append(f"  {key}:")
            for item in value[:5]:  # Limitar a 5 items
                if isinstance(item, dict):
                    lines.append(f"    - {', '.join(f'{k}: {v}' for k, v in item.items())}")
                else:
                    lines.append(f"    - {item}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def assemble_context(intent: str, df: pd.DataFrame) -> str:
    """Ensambla el bloque de contexto relevante segun la intencion detectada."""
    blocks = []

    if intent == "savings_advice":
        trends = compute_spending_trends(df)
        savings = compute_savings_rate(df)
        anomalies = detect_anomalies(df)
        recurring = compute_recurring_expenses(df)

        blocks.append(_format_analytics_block(trends, "TENDENCIAS DE GASTO"))
        blocks.append(_format_analytics_block(savings, "TASA DE AHORRO"))

        if anomalies:
            blocks.append("\nGASTOS ANOMALOS (inusualmente altos):")
            for a in anomalies[:5]:
                blocks.append(
                    f"  - {a['date']} | {a['description'][:50]} | "
                    f"{a['amount']:.2f} EUR (media {a['area']}: {a['mean']:.2f} EUR)"
                )

        if recurring:
            blocks.append("\nGASTOS RECURRENTES:")
            for r in recurring:
                blocks.append(
                    f"  - {r['name']}: ~{r['avg_amount']:.2f} EUR/mes "
                    f"({r['n_payments']} pagos, total {r['total']:.2f} EUR)"
                )

    elif intent == "concept_explanation":
        # Incluir transacciones de inversiones y resumen
        inv_summary = compute_investment_summary(df)
        blocks.append(_format_analytics_block(inv_summary, "RESUMEN DE INVERSIONES"))

        inv_transactions = get_transactions_by_area(df, "Investment", limit=15)
        blocks.append("\nTRANSACCIONES DE INVERSION:")
        blocks.append(format_transactions_for_prompt(inv_transactions))

        # Tambien incluir transacciones recientes para conceptos generales
        recent = get_recent_transactions(df, n=10)
        blocks.append("\nTRANSACCIONES RECIENTES:")
        blocks.append(format_transactions_for_prompt(recent))

    elif intent == "spending_query":
        summary = compute_monthly_summary(df)
        breakdown = compute_category_breakdown(df)
        blocks.append(_format_analytics_block(summary, "RESUMEN MES ACTUAL"))
        blocks.append(_format_analytics_block(breakdown, "DESGLOSE POR CATEGORIA"))

        top = get_top_expenses(df, n=10)
        blocks.append("\nMAYORES GASTOS:")
        blocks.append(format_transactions_for_prompt(top))

        recent = get_recent_transactions(df, n=15)
        blocks.append("\nTRANSACCIONES RECIENTES:")
        blocks.append(format_transactions_for_prompt(recent))

    else:  # general_chat
        recent = get_recent_transactions(df, n=10)
        blocks.append("TRANSACCIONES RECIENTES:")
        blocks.append(format_transactions_for_prompt(recent))

    return "\n".join(blocks)


def build_system_message(df: pd.DataFrame, intent: str) -> str:
    """Construye el system message completo con perfil y contexto relevante."""
    profile = build_user_profile(df)
    context = assemble_context(intent, df)

    return SYSTEM_PROMPT_TEMPLATE.format(
        profile=profile,
        context_block=f"DATOS RELEVANTES PARA ESTA CONSULTA:\n{context}",
    )
