# Práctica 4: Coach Financiero Generativo

## Descripción
Chatbot generativo que actúa como coach financiero personal. Analiza el historial de transacciones bancarias del usuario para ofrecer consejos de ahorro personalizados y explicar conceptos financieros complejos utilizando los propios gastos como ejemplos.

Desarrollado como cuarta práctica de la asignatura **Aplicaciones de Inteligencia Artificial (AIA)** del Grado en Ingeniería en Inteligencia Artificial de la Universidad Rey Juan Carlos.

### Características principales
- **Consejos de ahorro personalizados**: Analiza tendencias, anomalías y gastos recurrentes para dar recomendaciones concretas.
- **Explicación de conceptos financieros**: Usa las transacciones reales del usuario como ejemplos (interés compuesto, diversificación, etc.).
- **Consultas sobre gastos**: Responde preguntas sobre desglose por categoría, resúmenes mensuales y tendencias.
- **Interfaz de chat interactiva**: Aplicación Streamlit con métricas en sidebar, gráficos inline y conversación multi-turno.
- **Escalable**: Diseñado para manejar un historial de gastos que crece con el tiempo.

### Arquitectura
```
User Message (Spanish)
       |
  [Intent Router]        -- Clasificación por keywords (ahorro, concepto, gasto, general)
       |
  [Data Layer]           -- Carga CSV, parseo de importes EUR y fechas
       |
  [Analytics Engine]     -- Resúmenes, tendencias, anomalías, tasa de ahorro
       |
  [Structured Retrieval] -- Filtros pandas por Área/fecha/importe + búsqueda texto
       |
  [Context Assembly]     -- System prompt + perfil usuario + datos relevantes + historial
       |
  [Groq API]             -- LLM gratuito (Llama 3.3 70B por defecto)
       |
  [Streamlit UI]         -- Chat + métricas sidebar + gráficos inline
```

**Decisión de diseño**: Se usa retrieval estructurado con pandas en lugar de RAG con embeddings porque los datos son tabulares con columnas tipadas. Los filtros exactos son más precisos y eficientes que la búsqueda semántica a esta escala (~900 transacciones).

## Estructura del Proyecto

```text
.
├── app.py                  # Aplicación Streamlit (punto de entrada UI)
├── main.py                 # Punto de entrada CLI
├── requirements.txt        # Dependencias del proyecto
├── .env                    # Variables de entorno (GROQ_API_KEY)
├── src/
│   ├── data/
│   │   ├── loader.py       # Carga y parseo del CSV de transacciones
│   │   └── retrieval.py    # Filtros y búsqueda de transacciones
│   ├── features/
│   │   └── analytics.py    # Motor de analytics financiero
│   ├── models/
│   │   ├── coach.py        # Clase FinancialCoach (orquestador principal)
│   │   ├── intent.py       # Clasificador de intenciones por keywords
│   │   └── prompts.py      # Templates de prompts y ensamblaje de contexto
│   └── utils/
│       └── config.py       # Configuración centralizada
├── playground/
│   └── coach_prototype.ipynb  # Notebook de prototipado y comparación de modelos
├── tests/
│   ├── test_loader.py      # Tests del módulo de carga (8 tests)
│   └── test_analytics.py   # Tests del motor de analytics (17 tests)
├── doc/                    # Memoria LaTeX y documentación técnica
│   ├── sections/           # Capítulos de la memoria
│   └── main.tex            # Archivo LaTeX principal
├── data/                   # Datos (los datos fuente están en P2_AP-IA)
└── models/                 # Binarios de modelos entrenados
```

## Configuración del Entorno

### 1. Crear entorno virtual e instalar dependencias

```bash
# Crear entorno virtual con uv
uv venv .venv --python 3.12

# Instalar dependencias
uv pip install --link-mode=copy --python .venv/Scripts/python.exe -r requirements.txt
```

Si no tienes `uv`, puedes usar `pip` estándar:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar API Key de Groq

El proyecto usa **Groq API** (gratuita) para ejecutar modelos LLM open-source (Llama 3.3 70B).

1. Crear una cuenta gratuita en [console.groq.com](https://console.groq.com)
2. Ir a **API Keys** y crear una nueva key
3. Copiar la key (empieza por `gsk_...`) y pegarla en el archivo `.env`:

```env
GROQ_API_KEY=gsk_tu_clave_aqui
```

### 3. Datos de transacciones

Los datos financieros se cargan del CSV de la Práctica 2. Asegúrate de que existe el archivo:
```
../P2_AP-IA/data/raw/db_mod_descript.csv
```

## Ejecución

### Aplicación de chat (Streamlit)
```bash
.venv/Scripts/python.exe -m streamlit run app.py
```
Se abrirá en el navegador con:
- **Sidebar**: métricas del perfil financiero y gráfico de gastos por categoría
- **Chat**: interfaz conversacional con el coach financiero

### Notebook de prototipado
```bash
.venv/Scripts/python.exe -m jupyter notebook playground/coach_prototype.ipynb
```
Permite comparar respuestas de distintos modelos LLM y experimentar con prompts.

### Tests
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

## Modelo LLM

Tras comparar varios modelos en el notebook de prototipado, se seleccionó **Llama 3.3 70B** (`llama-3.3-70b-versatile`) por:
- Mejor calidad de respuesta en español
- Uso correcto de los datos financieros reales (no inventa cifras)
- Buena estructuración de consejos (headers, listas, cifras en EUR)
- Coherencia en conversaciones multi-turno

El modelo se puede cambiar desde el selector en la sidebar de la aplicación Streamlit.

## Autores
*   **Diego Esclarín** - (51729611N)
*   **Sofía Contreras** - (09848471D)

---
*Curso 2025-26 - Grado en Ingeniería en Inteligencia Artificial - URJC*
