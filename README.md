# Práctica 4: [TÍTULO DE LA PRÁCTICA]

## Descripción
Este repositorio contiene el desarrollo de la cuarta práctica de la asignatura **Aplicaciones de Inteligencia Artificial (AIA)** del Grado en Ingeniería en Inteligencia Artificial de la Universidad Rey Juan Carlos.

## Estructura del Proyecto
La estructura sigue los estándares de robustez y modularidad para entornos de producción:

```text
.
├── config/                 # Archivos de configuración (YAML, JSON, .env)
├── data/
│   ├── raw/                # Datos originales e inmutables
│   ├── processed/          # Datos limpios y transformados
│   └── external/           # Datos obtenidos de fuentes externas
├── doc/                    # Memoria LaTeX y documentación técnica
│   ├── figures/            # Imágenes utilizadas en la memoria
│   ├── sections/           # Capítulos de la memoria
│   └── main.tex            # Archivo LaTeX principal
├── logs/                   # Logs de ejecución
├── models/                 # Binarios de modelos entrenados
├── playground/             # Cuadernos Jupyter para pruebas rápidas
├── references/             # Papers y manuales de referencia
├── src/                    # Código fuente modular
│   ├── data/              # Scripts de carga y limpieza
│   ├── features/          # Ingeniería de variables
│   ├── models/            # Entrenamiento y arquitectura
│   ├── evaluation/        # Métricas y validación
│   └── pipeline.py        # Orquestador del flujo completo
├── tests/                  # Pruebas unitarias (Pytest)
├── .env                    # Variables de entorno
├── main.py                 # Punto de entrada principal
└── requirements.txt        # Dependencias del proyecto
```

## Configuración del Entorno
Se utiliza un entorno virtual compartido ubicado en el directorio superior:

```powershell
# Activar entorno virtual (Windows)
..\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución
Para ejecutar el pipeline completo de entrenamiento e inferencia:

```powershell
python main.py
```

## Autores
*   **Diego Esclarín** - (51729611N)
*   **Sofía Contreras** - (09848471D)

---
*Curso 2025-26 - Grado en Ingeniería en Inteligencia Artificial - URJC*
# P4_AP-IA
