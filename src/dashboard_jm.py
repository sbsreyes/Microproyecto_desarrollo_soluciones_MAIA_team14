"""
Dashboard Analítico — Customer Shopping Behavior

Descripción:
Aplicación interactiva desarrollada en Dash para la exploración visual y análisis
del comportamiento de compra de clientes. El dashboard permite filtrar dinámicamente
los datos y visualizar métricas clave de negocio para facilitar la toma de decisiones.

Objetivos:
- Analizar patrones de compra por categoría, edad y suscripción
- Identificar tendencias de consumo y segmentación de clientes
- Visualizar métricas clave (KPIs) en tiempo real
- Detectar relaciones entre variables relevantes del dataset

Funcionalidades principales:
- Filtros interactivos por categoría y rango de edad
- Indicadores KPI dinámicos
- Visualizaciones estadísticas y comparativas
- Análisis de dispersión, distribución y probabilidades
- Ranking de categorías por gasto promedio

Resultados esperados:
- Comprensión clara del comportamiento del cliente
- Insights accionables para negocio
- Base visual para reportes ejecutivos
- Soporte para futuras fases de modelado predictivo

Tecnologías:
Dash · Plotly · Pandas · Bootstrap · Python
"""
import dash
from dash import dcc,html,Input,Output,State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import requests

# --- CONFIG ---
API_URL="http://54.226.107.173:8001/api/v1/predict"

# --- APP ---
app=dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])

# --- FUNCION API ---
def call_api(payload):
    try:
        r=requests.post(API_URL,json=payload,timeout=30)
        if r.status_code!=200:
            return {"error":f"API respondió {r.status_code}","detail":r.text}
        return r.json()
    except Exception as e:
        return {"error":str(e)}

# --- LAYOUT ---
app.layout=dbc.Container([
    html.H1("📊 Predicción de Suscripción de Clientes",className="text-center my-4",style={"color":"white"}),

    dbc.Row([
        dbc.Col([dbc.Label("Edad"),dbc.Input(id="age",type="number",value=30)],width=4),
        dbc.Col([dbc.Label("Género"),
            dcc.Dropdown(id="gender",options=[
                {"label":"Masculino","value":"Male"},
                {"label":"Femenino","value":"Female"}],value="Male")],width=4),
        dbc.Col([dbc.Label("Categoría de Compra"),
            dcc.Dropdown(id="category",options=[
                {"label":"Ropa","value":"Clothing"},
                {"label":"Electrodomésticos","value":"Electrodomestic"},
                {"label":"Hogar","value":"Home"}],value="Clothing")],width=4)
    ],className="mb-3"),

    dbc.Row([
        dbc.Col([dbc.Label("Monto de Compra (USD)"),dbc.Input(id="purchase_amount",type="number",value=100)],width=4),
        dbc.Col([dbc.Label("¿Se aplicó descuento?"),
            dcc.Dropdown(id="discount",options=[
                {"label":"Sí","value":"Yes"},
                {"label":"No","value":"No"}],value="Yes")],width=4),
        dbc.Col([dbc.Label("Método de Pago"),
            dcc.Dropdown(id="payment_method",options=[
                {"label":"Tarjeta de Crédito","value":"Credit Card"},
                {"label":"Tarjeta Débito","value":"Debit Card"},
                {"label":"PayPal","value":"PayPal"}],value="Credit Card")],width=4)
    ],className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Button("🔎 Realizar Predicción",id="predict-button",color="success",size="lg",className="w-100"),width=6)
    ],justify="center",className="mb-4"),

    dbc.Row([
        dbc.Col(html.Div(id="prediction-output"),width=6)
    ],justify="center",className="mb-4")
    
],fluid=True,style={"backgroundColor":"#000000","minHeight":"100vh"})

# --- CALLBACK ---
@app.callback(
    Output("prediction-output","children"),
    Input("predict-button","n_clicks"),
    State("age","value"),
    State("gender","value"),
    State("category","value"),
    State("purchase_amount","value"),
    State("discount","value"),
    State("payment_method","value")
)
def make_prediction(n_clicks,age,gender,category,purchase_amount,discount,payment_method):
    if n_clicks is None:
        return ""

    payload={"inputs":[{
        "Age":age,
        "Purchase Amount (USD)":purchase_amount,
        "Review_Rating":4.0,
        "Previous_Purchases":5,
        "Gender":gender,
        "Category":category,
        "Location":"New York",
        "Size":"M",
        "Color":"Blue",
        "Season":"Summer",
        "Shipping Type":"Free Shipping",
        "Discount Applied":discount,
        "Payment Method":payment_method,
        "Frequency of Purchases":"Monthly",
        "Customer ID":1,
        "Item Purchased":"Shirt",
        "Promo Code Used":"No"
    }]}

    result=call_api(payload)

    if "error" in result:
        return dbc.Alert([html.H4("Error llamando la API"),html.P(str(result))],color="danger")

    pred=result["predictions"][0]
    label=pred["label"]
    prob=pred["probability"]
    color="success" if label=="Subscribed" else "danger"

    return dbc.Card(
        dbc.CardBody([
            html.H3("Resultado de la Predicción"),
            html.H4(label),
            html.P(f"Probabilidad: {round(prob*100,2)} %")
        ]),
        color=color,
        inverse=True
    )

# --- RUN (AWS EC2) ---
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8050,debug=False)