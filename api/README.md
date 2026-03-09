subscription-api/
│
├── app/                          ← Paquete principal de la API
│   ├── __init__.py
│   │
│   ├── main.py                   ← [PUNTO DE ENTRADA] Crea la app FastAPI,
│   │                                registra los routers, configura CORS y
│   │                                los eventos de startup/shutdown.
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py             ← [CONFIGURACIÓN] Variables de entorno de la API
│   │                                (host, puerto, nombre, versión, CORS origins).
│   │                                Centraliza todo lo configurable.
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── predict.py            ← [CONTRATOS DE DATOS] Define con Pydantic
│   │                                qué forma tienen los requests y responses.
│   │                                Si el JSON no cumple el esquema, FastAPI
│   │                                devuelve 422 automáticamente.
│   │
│   └── api/
│       ├── __init__.py
│       └── endpoints/
│           ├── __init__.py
│           ├── health.py         ← [ENDPOINT /health] Responde si la API
│           │                        está viva. Lo usa el dashboard y los
│           │                        monitores para saber si el servicio corre.
│           └── predict.py        ← [ENDPOINT /predict] Recibe los datos del
│                                    cliente, llama a make_prediction() del
│                                    paquete del modelo, devuelve el resultado.
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               ← Pruebas de integración de los endpoints
│
├── requirements.txt              ← Dependencias solo de la API (FastAPI, uvicorn)
└── run.py                        ← Script para arrancar el servidor localmente

## Flujo de una petición

Cliente (dashboard / curl)
    │
    │  POST /api/v1/predict
    │  { "inputs": [ { "Age": 35, "Gender": "Male", ... } ] }
    │
    ▼
FastAPI  →  schemas/predict.py  (valida el JSON)
    │
    ▼
api/endpoints/predict.py  (orquesta)
    │
    ▼
model.predict.make_prediction()  (del paquete instalado)
    │
    ├── ShoppingPreprocessor.transform()
    ├── StandardScaler.transform()
    └── LogisticRegression.predict_proba()
    │
    ▼
PredictionResponse { predictions: [1], probabilities: [0.81], version: "0.0.1" }
    │
    ▼
Cliente recibe JSON