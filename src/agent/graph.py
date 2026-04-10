"""Grafo LangGraph del coach financiero con memoria persistente."""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.memory import load_memory, save_memory
from src.agent.state import AgentState
from src.agent.tools import (
    ALL_TOOLS,
    get_goals,
    set_dataframe,
    set_goals,
)
from src.features.analytics import compute_monthly_summary
from src.models.prompts import build_user_profile
from src.utils.config import (
    KEEP_RECENT_MESSAGES,
    SUMMARIZATION_THRESHOLD,
    SUMMARY_MAX_TOKENS,
)

import pandas as pd


SYSTEM_PROMPT = """\
Eres un coach financiero personal experto y amable. Siempre respondes en espanol.

REGLAS ESTRICTAS:
- Basa tus respuestas EXCLUSIVAMENTE en los datos financieros reales del usuario.
- NUNCA inventes transacciones, cifras o datos.
- Usa las herramientas disponibles para consultar datos antes de responder.
- Se motivador y practico. Da consejos especificos y accionables.
- Formatea cantidades en EUR con dos decimales.
- Si no tienes datos suficientes para responder, dilo honestamente.
- Responde de forma concisa pero completa.

PERFIL FINANCIERO DEL USUARIO:
{profile}

OBJETIVOS ACTIVOS:
{goals}

ALERTAS ACTUALES:
{alerts}
{summary_section}"""


SUMMARY_SYSTEM_PROMPT = """\
Eres un asistente que resume conversaciones entre un usuario y su coach financiero.

Tu tarea: producir un resumen conciso (maximo 200 palabras) que capture:
- Temas tratados (ahorro, gastos, inversiones, conceptos financieros, objetivos...)
- Datos financieros relevantes mencionados (cifras, categorias, periodos)
- Decisiones tomadas o consejos dados
- Objetivos definidos o modificados
- Preferencias del usuario que hayas detectado

Escribe el resumen en espanol, en tercera persona, en parrafos cortos.
Si hay un resumen previo, integralo con la nueva informacion sin duplicar."""


def build_graph(
    df: pd.DataFrame,
    api_key: str,
    model: str,
    user_id: str = "default",
) -> StateGraph:
    """Construye y compila el grafo del agente financiero."""

    # Inyectar datos en las tools
    set_dataframe(df)

    # Perfil precalculado (se usa en el system prompt)
    profile_text = build_user_profile(df)

    # LLM con tools bindeadas (para el agente principal)
    llm = ChatGroq(api_key=api_key, model=model, temperature=0.5)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # LLM dedicado a resumir (sin tools, mas determinista, max_tokens limitado)
    summarizer_llm = ChatGroq(api_key=api_key, model=model, temperature=0.2, max_tokens=SUMMARY_MAX_TOKENS)

    # ── Nodos ────────────────────────────────────────────────────────

    def load_memory_node(state: AgentState) -> dict:
        """Carga memoria persistente y sincroniza goals con las tools."""
        memory = load_memory(user_id)
        set_goals(memory.get("goals", []))
        return {
            "memory": memory,
            "current_alerts": [],
        }

    def summarize_node(state: AgentState) -> dict:
        """Resume conversaciones largas para no superar la ventana de contexto.

        Si state['messages'] supera SUMMARIZATION_THRESHOLD, toma todos los mensajes
        excepto los KEEP_RECENT_MESSAGES mas recientes, los pasa por un LLM resumidor
        y los reemplaza por un unico resumen guardado en memory['summary'].
        """
        messages = state["messages"]
        if len(messages) <= SUMMARIZATION_THRESHOLD:
            return {}  # noop si no se ha alcanzado el umbral

        memory = dict(state.get("memory", {}))
        previous_summary = memory.get("summary", "")

        # Mensajes a resumir: todos menos los KEEP_RECENT mas recientes
        to_summarize = messages[:-KEEP_RECENT_MESSAGES]

        # Convertir mensajes a texto plano
        conv_lines = []
        for m in to_summarize:
            content = getattr(m, "content", None)
            if not content:
                continue
            role = m.__class__.__name__.replace("Message", "")
            conv_lines.append(f"{role}: {content}")
        conv_text = "\n".join(conv_lines)

        # Construir prompt: si ya hay resumen previo, integrarlo
        if previous_summary:
            user_content = (
                f"RESUMEN PREVIO DE LA CONVERSACION:\n{previous_summary}\n\n"
                f"NUEVOS MENSAJES A INTEGRAR:\n{conv_text}\n\n"
                "Produce un resumen actualizado que combine ambos."
            )
        else:
            user_content = (
                f"Resume la siguiente conversacion:\n\n{conv_text}"
            )

        summary_response = summarizer_llm.invoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        new_summary = summary_response.content

        # Persistir el nuevo resumen en memoria
        memory["summary"] = new_summary

        # Eliminar los mensajes resumidos del estado via RemoveMessage
        delete_msgs = [
            RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None)
        ]

        return {
            "messages": delete_msgs,
            "memory": memory,
        }

    def agent_node(state: AgentState) -> dict:
        """Invoca al LLM con el system prompt, historial y tools."""
        memory = state.get("memory", {})
        goals = memory.get("goals", [])
        alerts = state.get("current_alerts", [])
        summary = memory.get("summary", "")

        # Formatear goals para el prompt
        if goals:
            goals_text = "\n".join(
                f"- {g['category']}: max {g['limit']:.2f} EUR/mes (hasta {g['deadline']})"
                for g in goals
            )
        else:
            goals_text = "Sin objetivos definidos."

        alerts_text = "\n".join(f"- {a}" for a in alerts) if alerts else "Sin alertas."

        # Bloque de resumen: solo si existe
        if summary:
            summary_section = f"\nRESUMEN DE CONVERSACIONES PREVIAS:\n{summary}\n"
        else:
            summary_section = ""

        sys_msg = SystemMessage(content=SYSTEM_PROMPT.format(
            profile=profile_text,
            goals=goals_text,
            alerts=alerts_text,
            summary_section=summary_section,
        ))

        # Ensamblar mensajes: system + historial de conversacion
        messages = [sys_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    def check_goals_node(state: AgentState) -> dict:
        """Evalua todos los goals y genera alertas si se supera el 80%."""
        memory = state.get("memory", {})
        goals = memory.get("goals", [])
        past_alerts = set(memory.get("past_alerts", []))
        new_alerts: list[str] = []

        now = datetime.now()
        for goal in goals:
            summary = compute_monthly_summary(df, now.year, now.month)
            spent = summary["breakdown"].get(goal["category"], 0)
            limit = goal["limit"]
            if limit <= 0:
                continue
            pct = spent / limit * 100
            alert_key = f"{goal['category']}_{now.strftime('%Y-%m')}_{int(pct // 10) * 10}"

            if pct >= 100 and alert_key not in past_alerts:
                msg = (
                    f"Has SUPERADO tu objetivo en {goal['category']}: "
                    f"{spent:.2f} / {limit:.2f} EUR ({pct:.0f}%)"
                )
                new_alerts.append(msg)
                past_alerts.add(alert_key)
            elif pct >= 80 and alert_key not in past_alerts:
                msg = (
                    f"Atencion: llevas {spent:.2f} de {limit:.2f} EUR "
                    f"en {goal['category']} ({pct:.0f}%)"
                )
                new_alerts.append(msg)
                past_alerts.add(alert_key)

        memory["past_alerts"] = list(past_alerts)
        return {"memory": memory, "current_alerts": new_alerts}

    def save_memory_node(state: AgentState) -> dict:
        """Sincroniza goals de las tools al estado y persiste en disco."""
        memory = state.get("memory", {})
        # Los goals pueden haber cambiado via set_goal/remove_goal tools
        memory["goals"] = get_goals()
        save_memory(memory, user_id)
        return {"memory": memory}

    # ── Edges condicionales ──────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        """Si el ultimo mensaje del LLM tiene tool_calls, ir a tools. Si no, a check_goals."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "check_goals"

    # ── Ensamblar grafo ──────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_goals", check_goals_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "check_goals": "check_goals",
    })
    graph.add_edge("tools", "agent")  # Vuelve al agente tras ejecutar tool
    graph.add_edge("check_goals", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
