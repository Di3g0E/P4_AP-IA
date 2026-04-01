"""Coach Financiero Generativo - Aplicacion Streamlit."""

import streamlit as st
import pandas as pd

from src.data.loader import load_transactions
from src.models.coach import FinancialCoach
from src.models.prompts import build_user_profile
from src.features.analytics import (
    compute_category_breakdown,
    compute_savings_rate,
)
from src.utils.config import DATA_PATH, GROQ_API_KEY, DEFAULT_MODEL


# --- Configuracion de pagina ---
st.set_page_config(
    page_title="Coach Financiero",
    page_icon="💰",
    layout="wide",
)


# --- Inicializacion de estado ---
@st.cache_data
def load_data():
    return load_transactions(DATA_PATH)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "coach" not in st.session_state:
        st.session_state.coach = FinancialCoach(
            api_key=GROQ_API_KEY,
            model=st.session_state.get("selected_model", DEFAULT_MODEL),
        )


# --- Sidebar: perfil financiero ---
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.header("Tu Perfil Financiero")

        expenses = df[df["Type"] == "Expenses"]
        income = df[df["Type"] == "Income"]
        total_inc = income["Amount_clean"].sum()
        total_exp = expenses["Amount_clean"].sum()

        col1, col2 = st.columns(2)
        col1.metric("Ingresos", f"{total_inc:,.0f} EUR")
        col2.metric("Gastos", f"{total_exp:,.0f} EUR")

        net = total_inc - total_exp
        savings_pct = (net / total_inc * 100) if total_inc > 0 else 0
        col3, col4 = st.columns(2)
        col3.metric("Ahorro Neto", f"{net:,.0f} EUR")
        col4.metric("Tasa Ahorro", f"{savings_pct:.1f}%")

        st.metric("Transacciones", len(df))
        st.caption(
            f"{df['Date_parsed'].min().strftime('%d/%m/%Y')} - "
            f"{df['Date_parsed'].max().strftime('%d/%m/%Y')}"
        )

        # Grafico de gastos por categoria
        st.subheader("Gastos por Categoria")
        breakdown = compute_category_breakdown(df)
        chart_df = pd.DataFrame([
            {"Categoria": cat, "EUR": info["total"]}
            for cat, info in breakdown["categories"].items()
        ]).set_index("Categoria")
        st.bar_chart(chart_df)

        # Selector de modelo
        st.divider()
        models = {
            "Llama 3.3 70B": "llama-3.3-70b-versatile",
            "Llama 3.1 8B": "llama-3.1-8b-instant",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Gemma 2 9B": "gemma2-9b-it",
        }
        selected = st.selectbox(
            "Modelo LLM",
            options=list(models.keys()),
            index=0,
        )
        new_model = models[selected]
        if st.session_state.get("selected_model") != new_model:
            st.session_state.selected_model = new_model
            st.session_state.coach = FinancialCoach(
                api_key=GROQ_API_KEY, model=new_model, df=df
            )

        # Boton para recargar datos
        if st.button("Recargar datos"):
            st.cache_data.clear()
            st.rerun()


# --- Chat principal ---
def render_chat(df: pd.DataFrame):
    st.title("Coach Financiero")
    st.caption("Tu asistente de finanzas personales con IA")

    # Mensaje de bienvenida
    if not st.session_state.messages:
        expenses = df[df["Type"] == "Expenses"]
        income = df[df["Type"] == "Income"]
        total_inc = income["Amount_clean"].sum()
        total_exp = expenses["Amount_clean"].sum()
        savings_pct = (total_inc - total_exp) / total_inc * 100 if total_inc > 0 else 0

        welcome = (
            f"Hola! Soy tu coach financiero personal. "
            f"He analizado tus **{len(df)} transacciones** desde "
            f"{df['Date_parsed'].min().strftime('%B %Y')} hasta "
            f"{df['Date_parsed'].max().strftime('%B %Y')}.\n\n"
            f"Tu tasa de ahorro media es del **{savings_pct:.1f}%**. "
            f"Preguntame lo que quieras sobre tus finanzas:\n\n"
            f"- *Dame consejos para ahorrar mas*\n"
            f"- *Cuanto gasto al mes en ocio?*\n"
            f"- *Que es el interes compuesto?*\n"
            f"- *Analiza mis tendencias de gasto*"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    # Renderizar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre finanzas..."):
        # Mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando tus finanzas..."):
                # Preparar historial (sin el system message, solo user/assistant)
                history = [
                    m for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]

                response = st.session_state.coach.chat(
                    message=prompt,
                    conversation_history=history,
                )

                st.markdown(response.text)

                # Mostrar grafico si aplica
                if response.chart_data is not None and response.chart_type == "bar":
                    st.bar_chart(
                        response.chart_data.set_index("Categoria")["Total EUR"]
                    )

        st.session_state.messages.append(
            {"role": "assistant", "content": response.text}
        )


# --- Main ---
def main():
    df = load_data()
    init_session()
    render_sidebar(df)
    render_chat(df)


if __name__ == "__main__":
    main()
