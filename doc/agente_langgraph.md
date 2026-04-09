# Agente LangGraph con Memoria Persistente — Documentacion tecnica

Documento de referencia para la memoria del proyecto. Describe la implementacion del agente LangGraph, los cambios respecto a la version clasica y la comparativa entre ambas arquitecturas.

---

## 1. Arquitectura anterior (Coach clasico)

### Flujo
```
User Message → Intent Router (keywords) → Context Assembly (datos fijos) → Groq API → Respuesta
```

### Funcionamiento
- El mensaje del usuario se clasifica por keywords en 4 intenciones: `savings_advice`, `concept_explanation`, `spending_query`, `general_chat`.
- Segun la intencion, un bloque `if/elif` en `prompts.py` decide que funciones de analytics llamar y ensambla un contexto estatico.
- El contexto completo (perfil + datos relevantes + historial) se envia al LLM en una sola llamada.
- El LLM genera la respuesta sin acceso directo a los datos; solo ve lo que el codigo decidio incluir en el prompt.

### Limitaciones
- **Rigido**: la clasificacion por keywords falla con frases ambiguas o preguntas compuestas.
- **Sin encadenamiento**: si la respuesta necesita datos de dos categorias distintas, el intent router solo elige una rama.
- **Sin estado entre sesiones**: cada conversacion empieza desde cero.
- **Contexto fijo**: el LLM recibe siempre el mismo bloque de datos por intent, aunque la pregunta solo necesite una parte.

---

## 2. Arquitectura nueva (Agente LangGraph)

### Flujo
```
User Message → Load Memory → Agent (LLM + tools) ⇄ Tool Node → Check Goals → Save Memory → Respuesta
```

### Grafo LangGraph (5 nodos)

| Nodo | Funcion |
|---|---|
| `load_memory` | Carga objetivos y alertas del usuario desde JSON en disco |
| `agent` | Invoca al LLM con system prompt + historial + 12 tools bindeadas |
| `tools` | Ejecuta la tool que el LLM eligio (condicional, solo si hay tool_call) |
| `check_goals` | Evalua todos los objetivos activos y genera alertas si se supera el 80% |
| `save_memory` | Persiste el estado actualizado (goals, alerts) a disco |

### Edge condicional (bucle ReAct)
- Si el ultimo mensaje del LLM contiene `tool_calls` → va al nodo `tools` → vuelve al nodo `agent`.
- Si no contiene `tool_calls` → va a `check_goals` → `save_memory` → END.
- El agente puede encadenar multiples tools antes de responder.

### Tools disponibles (12)

**Analytics (8 wrappers de funciones existentes):**

| Tool | Funcion que envuelve | Descripcion |
|---|---|---|
| `monthly_summary` | `compute_monthly_summary()` | Resumen de un mes: ingresos, gastos, ahorro, desglose |
| `spending_trends` | `compute_spending_trends()` | Tendencias de gasto mensual y por categoria |
| `category_breakdown` | `compute_category_breakdown()` | Desglose por categoria con totales y porcentajes |
| `savings_rate` | `compute_savings_rate()` | Tasa de ahorro (ingreso - gasto) / ingreso |
| `anomalies` | `detect_anomalies()` | Gastos anomalamente altos por categoria |
| `recurring_expenses` | `compute_recurring_expenses()` | Gastos recurrentes (suscripciones) |
| `top_expenses` | `get_top_expenses()` | N mayores gastos, opcionalmente filtrados por area |
| `search_by_description` | `search_transactions()` | Busqueda por texto en descripciones |

**Objetivos (4 tools nuevas):**

| Tool | Descripcion |
|---|---|
| `set_goal` | Establece un objetivo de gasto maximo mensual para una categoria |
| `check_goal_progress` | Comprueba gasto actual vs limite y devuelve porcentaje |
| `list_goals` | Lista todos los objetivos activos del usuario |
| `remove_goal` | Elimina un objetivo de una categoria |

### Memoria persistente

**Estructura del JSON (`memory/{user_id}.json`):**
```json
{
  "goals": [
    {"category": "Ocio", "limit": 200.0, "deadline": "2026-06", "created_at": "2026-04-08"}
  ],
  "past_alerts": ["Ocio_2026-04_80"],
  "savings_tips_given": []
}
```

- **goals**: objetivos de ahorro activos. Se crean/eliminan via tools `set_goal`/`remove_goal`.
- **past_alerts**: claves de alertas ya enviadas (`{category}_{YYYY-MM}_{umbral}`). Evita repetir la misma alerta en la misma sesion o en sesiones posteriores del mismo mes.
- **savings_tips_given**: reservado para registrar consejos ya dados y evitar repeticiones (sin uso activo actualmente).

### Sistema de alertas

El nodo `check_goals` se ejecuta tras cada respuesta del agente:
1. Para cada goal activo, calcula el gasto del mes actual en esa categoria.
2. Si el porcentaje consumido supera el 80% o el 100%, genera una alerta.
3. Solo genera la alerta si la clave correspondiente no existe en `past_alerts`.
4. Las alertas se inyectan al final de la respuesta del agente.

---

## 3. Comparativa directa

| Aspecto | Coach clasico | Agente LangGraph |
|---|---|---|
| **Quien decide que datos consultar** | El codigo (intent router + if/elif) | El LLM (elige tools dinamicamente) |
| **Flujo de datos** | Pipeline lineal, una sola pasada | Bucle ReAct, puede encadenar multiples tools |
| **Preguntas compuestas** | Cae en un solo intent, contexto parcial | El agente llama a varias tools y combina resultados |
| **Estado entre sesiones** | Sin estado, cada sesion empieza vacia | Memoria persistente (objetivos, alertas) en JSON |
| **Objetivos de ahorro** | No soportado | El usuario define limites y el agente monitoriza |
| **Alertas** | No soportado | Automaticas cuando un objetivo supera el 80% |
| **Contexto enviado al LLM** | Bloque fijo segun intent (puede ser excesivo o insuficiente) | Solo los datos que el LLM pidio via tools (mas preciso) |
| **Latencia** | Una sola llamada al LLM | Multiples llamadas (1 por tool + respuesta final) |
| **Robustez ante frases ambiguas** | Depende de keywords exactos | El LLM interpreta la intencion semanticamente |
| **Fallback** | No necesario (unica opcion) | Toggle en sidebar para volver al coach clasico |

---

## 4. Archivos implicados

### Archivos nuevos
| Archivo | Responsabilidad |
|---|---|
| `src/agent/__init__.py` | Declaracion del modulo |
| `src/agent/state.py` | `AgentState`, `Goal`, `UserMemory` |
| `src/agent/tools.py` | 12 tools (8 wrappers + 4 de goals) |
| `src/agent/memory.py` | `load_memory` / `save_memory` |
| `src/agent/graph.py` | Grafo LangGraph con 5 nodos |
| `src/models/coach_agent.py` | Clase `AgentCoach` (wrapper del grafo) |
| `tests/test_agent_tools.py` | 18 tests de tools |
| `tests/test_memory.py` | 6 tests de memoria persistente |

### Archivos modificados
| Archivo | Cambio |
|---|---|
| `requirements.txt` | +`langgraph`, `langchain-groq`, `langchain-core` |
| `app.py` | Toggle agente/clasico, seccion objetivos en sidebar, soporte dual AgentCoach/FinancialCoach |
| `README.md` | Arquitectura del agente, tabla de tools, estructura actualizada |

### Archivos sin cambios
`coach.py`, `intent.py`, `prompts.py`, `analytics.py`, `retrieval.py`, `loader.py`, `config.py` — la logica de negocio original se mantiene intacta y se reutiliza como tools.

---

## 5. Tests

**49 tests totales (todos passing):**
- `test_loader.py`: 8 tests — carga y parseo del CSV
- `test_analytics.py`: 17 tests — funciones de analytics
- `test_agent_tools.py`: 18 tests — 8 tools de analytics + 8 de goals + 2 de estructura
- `test_memory.py`: 6 tests — load, save, roundtrip, directorios, multiusuario

---

## 6. Ejemplo de interaccion con el agente

### Sesion 1: definir objetivo
```
Usuario: "Quiero no gastar mas de 150 EUR en restaurantes este mes"

load_memory   → goals: [] (primera vez)
agent         → tool_call: set_goal("Restauracion", 150, "2026-04")
tools         → registra goal en estado
agent         → tool_call: check_goal_progress("Restauracion")
tools         → gasto actual: 89 EUR / 150 EUR (59%)
agent         → responde con contexto
check_goals   → 59% < 80%, sin alerta
save_memory   → goals: [{"category": "Restauracion", "limit": 150, ...}]

Respuesta: "Objetivo registrado. Llevas 89 EUR de 150 EUR en restaurantes
            este mes (59%). Te avisare si te acercas al limite."
```

### Sesion 2: seguimiento automatico
```
Usuario: "Como van mis finanzas este mes?"

load_memory   → goals: [{"category": "Restauracion", "limit": 150, ...}]
agent         → tool_call: monthly_summary(2026, 4)
tools         → resumen del mes
agent         → tool_call: spending_trends(3)
tools         → tendencias ultimos 3 meses
agent         → responde con ambos resultados
check_goals   → Restauracion: 142 / 150 EUR (95%) → ALERTA NUEVA
save_memory   → past_alerts: ["Restauracion_2026-04_90"]

Respuesta: "Este mes llevas X EUR en gastos totales...
            ---
            Alertas de objetivos:
            - Atencion: llevas 142 de 150 EUR en Restauracion (95%)"
```
