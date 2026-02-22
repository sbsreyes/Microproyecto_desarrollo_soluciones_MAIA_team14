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
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


# 1. Cargar datos
df = pd.read_csv('data/raw/shopping_behavior_updated.csv')


# 2. Estilo visual profesional
STYLE = {
    'bg': '#0B0F14',
    'card': '#121821',
    'accent': '#22C55E',
    'accent_soft': '#4ADE80',
    'text': '#E6EDF3',
    'muted': '#94A3B8',
    'grid': '#1E293B'
}

external_stylesheets = [
    dbc.themes.CYBORG,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)


# KPI COMPONENT
def kpi_box(label, id):
    return html.Div([
        html.P(label, style={
            'color': STYLE['muted'],
            'fontSize': '10px',
            'marginBottom': '2px',
            'letterSpacing': '1px'
        }),
        html.H3(id=id, style={
            'color': STYLE['accent'],
            'fontWeight': '600',
            'margin': '0px'
        })
    ], style={
        'padding': '8px 14px',
        'borderRight': f'1px solid {STYLE["grid"]}'
    })


# LAYOUT
app.layout = html.Div(
    style={
        'backgroundColor': STYLE['bg'],
        'minHeight': '100vh',
        'padding': '25px',
        'fontFamily': 'Inter, Segoe UI'
    },
    children=[
        dbc.Container([

            # HEADER
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("INSIGHTS ENGINE // BEHAVIOR_ANALYSIS",
                            style={'letterSpacing': '2px', 'fontWeight': '600', 'display': 'inline'}),
                    html.Span("LIVE", style={
                        "background": STYLE["accent"],
                        "color": "black",
                        "padding": "2px 8px",
                        "borderRadius": "6px",
                        "fontSize": "10px",
                        "marginLeft": "10px"
                    })
                ]), md=6),

                dbc.Col(html.Div([
                    kpi_box("RECORDS", "kpi-total"),
                    kpi_box("AVG_USD", "kpi-avg"),
                    kpi_box("SUB_%", "kpi-sub"),
                ], className="d-flex justify-content-end"), md=6)
            ], className="mb-4 align-items-center"),


            dbc.Row([

                # SIDEBAR
                dbc.Col([
                    html.Div([

                        html.Label("CATEGORY_SELECT",
                                   style={'fontSize': '10px', 'color': STYLE['muted']}),

                        dcc.Dropdown(
                            id='category-filter',
                            options=[{'label': c, 'value': c} for c in df['Category'].unique()]
                                    + [{'label': 'GLOBAL', 'value': 'all'}],
                            value='all',
                            className="mb-3",
                            style={'fontSize': '12px'}
                        ),

                        html.Label("AGE_PARAMETER",
                                   style={'fontSize': '10px', 'color': STYLE['muted']}),

                        dcc.RangeSlider(
                            id='age-slider',
                            min=df['Age'].min(),
                            max=df['Age'].max(),
                            value=[df['Age'].min(), df['Age'].max()],
                            marks={i: {'label': str(i),
                                       'style': {'color': STYLE['muted'], 'fontSize': '10px'}}
                                   for i in range(20, 71, 10)}
                        ),

                    ], style={
                        'padding': '15px',
                        'backgroundColor': STYLE['card'],
                        'borderRadius': '8px'
                    })
                ], md=3),


                # PANEL GRÁFICOS
                dbc.Col([

                    dbc.Row([
                        dbc.Col(dcc.Graph(id='subscription-plot',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),

                        dbc.Col(dcc.Graph(id='amount-dist-plot',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Col(dcc.Graph(id='discount-heatmap',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),

                        dbc.Col(dcc.Graph(id='age-scatter-plot',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Col(dcc.Graph(id='category-avg-plot',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),

                        dbc.Col(dcc.Graph(id='subscription-box-plot',
                                          config={'displayModeBar': False},
                                          style={'height': '290px'}), md=6),
                    ], className="g-2")

                ], md=9)
            ])
        ], fluid=True)
    ]
)


# CALLBACK
@app.callback(
    [
        Output('subscription-plot', 'figure'),
        Output('amount-dist-plot', 'figure'),
        Output('discount-heatmap', 'figure'),
        Output('age-scatter-plot', 'figure'),
        Output('category-avg-plot', 'figure'),
        Output('subscription-box-plot', 'figure'),
        Output('kpi-total', 'children'),
        Output('kpi-avg', 'children'),
        Output('kpi-sub', 'children')
    ],
    [Input('category-filter', 'value'),
     Input('age-slider', 'value')]
)
def update_dashboard(cat, age):

    dff = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]
    if cat != 'all':
        dff = dff[dff['Category'] == cat]


    # KPIs
    kpi1 = f"{len(dff)}"
    kpi2 = f"{dff['Purchase Amount (USD)'].mean():.1f}"
    kpi3 = f"{(dff['Subscription Status'] == 'Yes').mean():.0%}"


    # Layout global gráficos
    layout_cfg = {
        'template': 'plotly_dark',
        'paper_bgcolor': STYLE['card'],
        'plot_bgcolor': STYLE['card'],
        'margin': dict(t=35, b=30, l=40, r=20),
        'font': {'size': 11, 'color': STYLE['text'], 'family': 'Inter'},
        'title': {'font': {'size': 13, 'color': STYLE['accent']}, 'y': 0.95},
        'xaxis': {'gridcolor': STYLE['grid'], 'zeroline': False},
        'yaxis': {'gridcolor': STYLE['grid'], 'zeroline': False},
        'transition_duration': 400
    }


    # 1 Suscripción por género
    f1 = px.histogram(
        dff,
        x="Gender",
        color="Subscription Status",
        barmode="group",
        color_discrete_sequence=[STYLE['accent'], STYLE['accent_soft']],
        title="SUBSCRIPTION BY GENDER"
    )
    f1.update_traces(marker_line_width=0)


    # 2 Distribución compras
    f2 = px.histogram(
        dff,
        x="Purchase Amount (USD)",
        color_discrete_sequence=[STYLE['accent']],
        title="PURCHASE INTENSITY"
    )


    # 3 Heatmap
    ct = pd.crosstab(
        dff['Discount Applied'],
        dff['Subscription Status'],
        normalize='index'
    )

    f3 = px.imshow(
        ct,
        text_auto=".2f",
        color_continuous_scale=[STYLE['bg'], STYLE['accent']],
        title="PROBABILITY MATRIX"
    )
    f3.update_layout(coloraxis_showscale=False)


    # 4 Scatter
    f4 = px.scatter(
        dff,
        x="Age",
        y="Purchase Amount (USD)",
        color_discrete_sequence=[STYLE['accent']],
        opacity=0.35,
        title="AGE DISPERSION"
    )
    f4.update_traces(marker=dict(size=5))


    # 5 Ranking categoría
    cat_avg = (
        dff.groupby("Category")["Purchase Amount (USD)"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    f5 = px.bar(
        cat_avg,
        x="Purchase Amount (USD)",
        y="Category",
        orientation="h",
        color_discrete_sequence=[STYLE['accent']],
        title="AVG PURCHASE BY CATEGORY"
    )


    # 6 Boxplot suscripción
    f6 = px.box(
        dff,
        x="Subscription Status",
        y="Purchase Amount (USD)",
        color="Subscription Status",
        color_discrete_sequence=[STYLE['accent'], STYLE['accent_soft']],
        title="SPENDING BEHAVIOR BY SUBSCRIPTION"
    )


    # aplicar layout global
    for f in [f1, f2, f3, f4, f5, f6]:
        f.update_layout(layout_cfg)


    return f1, f2, f3, f4, f5, f6, kpi1, kpi2, kpi3


if __name__ == '__main__':
    app.run(debug=True)