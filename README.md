# Microproyecto_desarrollo_soluciones_MAIA_team14
Proyecto de Analítica de datos e inteligencia artificial que busca predecir el comportamiento de clientes en sus hábitos de compra, basado en datos demográficos, transaccionales y de comportamiento. Incluye API de inferencia y dashboard de monitoreo. Desarrollado como producto de datos end-to-end.

Alcance del Proyecto
El entregable principal es un modelo machine learning que ... a partir de variables demográficas, transaccionales y de comportamiento. El sistema incluye un pipeline de ML entrenado, una API de inferencia y un dashboard de monitoreo, cubriendo el ciclo de vida completo desde los datos crudos hasta predicciones listas para producción.
Stack Tecnológico

ML y Experimentación: scikit-learn, MLflow
API: FastAPI
Dashboard: Streamlit / Plotly
Versionamiento: Git (código), DVC (datos y modelos)
Lenguaje: Python 3.11+

## Estructura del Proyecto
```
├── data/               # Datasets crudos y procesados (rastreados con DVC)
├── notebooks/          # EDA y experimentación
├── src/                # Código fuente (features, entrenamiento, inferencia)
├── api/                # Capa de servicio con FastAPI
├── dashboard/          # Monitoreo y visualización
├── models/             # Modelos serializados y artefactos
├── mlflow/             # Seguimiento de experimentos
└── tests/              # Pruebas unitarias y de integración
```
