import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# 1. Cargar datos
df = pd.read_csv('data/raw/shopping_behavior_updated.csv')

# 2. Configuración Estética (Executive Dark Theme)
STYLE = {
    'bg': '#0A0A0B',
    'card': '#141416',
    'accent': '#27AE60',  # Verde Esmeralda (Sofisticado)
    'text': '#E1E1E1',
    'muted': '#888888'
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# --- COMPONENTES ---
def kpi_box(label, id):
    return html.Div([
        html.P(label, style={'color': STYLE['muted'], 'fontSize': '11px', 'marginBottom': '0px', 'textTransform': 'uppercase'}),
        html.H4(id=id, style={'color': STYLE['accent'], 'fontWeight': '600', 'marginBottom': '0px'})
    ], style={'padding': '10px', 'borderRight': '1px solid #222'})

# --- LAYOUT ---
app.layout = html.Div(style={'backgroundColor': STYLE['bg'], 'minHeight': '100vh', 'padding': '25px', 'fontFamily': 'Segoe UI, Roboto'}, children=[
    dbc.Container([
        # Header compacto
        dbc.Row([
            dbc.Col(html.H5("INSIGHTS ENGINE // BEHAVIOR_ANALYSIS", style={'letterSpacing': '2px', 'fontWeight': 'bold'}), md=6),
            dbc.Col(html.Div([
                kpi_box("RECORDS", "kpi-total"),
                kpi_box("AVG_USD", "kpi-avg"),
                kpi_box("SUB_%", "kpi-sub"),
            ], className="d-flex justify-content-end"), md=6)
        ], className="mb-4 align-items-center"),

        dbc.Row([
            # Lateral: Filtros pequeños
            dbc.Col([
                html.Div([
                    html.Label("CATEGORY_SELECT", style={'fontSize': '10px', 'color': STYLE['muted']}),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': c, 'value': c} for c in df['Category'].unique()] + [{'label': 'GLOBAL', 'value': 'all'}],
                        value='all', className="mb-3", style={'fontSize': '12px'}
                    ),
                    html.Label("AGE_PARAMETER", style={'fontSize': '10px', 'color': STYLE['muted']}),
                    dcc.RangeSlider(
                        id='age-slider', min=df['Age'].min(), max=df['Age'].max(),
                        value=[df['Age'].min(), df['Age'].max()],
                        marks={i: {'label': str(i), 'style': {'color': STYLE['muted'], 'fontSize': '10px'}} for i in range(20, 71, 10)},
                    ),
                ], style={'padding': '15px', 'backgroundColor': STYLE['card'], 'borderRadius': '4px'})
            ], md=3),

            # Grid de gráficos compactos
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='subscription-plot', style={'height': '280px'}), md=6),
                    dbc.Col(dcc.Graph(id='amount-dist-plot', style={'height': '280px'}), md=6),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='discount-heatmap', style={'height': '280px'}), md=6),
                    dbc.Col(dcc.Graph(id='age-scatter-plot', style={'height': '280px'}), md=6),
                ], className="g-2"),
            ], md=9)
        ])
    ], fluid=True)
])

# --- LÓGICA ---
@app.callback(
    [Output('subscription-plot', 'figure'), Output('amount-dist-plot', 'figure'),
     Output('discount-heatmap', 'figure'), Output('age-scatter-plot', 'figure'),
     Output('kpi-total', 'children'), Output('kpi-avg', 'children'), Output('kpi-sub', 'children')],
    [Input('category-filter', 'value'), Input('age-slider', 'value')]
)
def update_dashboard(cat, age):
    dff = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]
    if cat != 'all': dff = dff[dff['Category'] == cat]

    # KPIs
    kpi1 = f"{len(dff)}"
    kpi2 = f"{dff['Purchase Amount (USD)'].mean():.1f}"
    kpi3 = f"{(dff['Subscription Status'] == 'Yes').mean():.0%}"

    # Estética de gráficos
    layout_cfg = {
        'template': 'plotly_dark', 'paper_bgcolor': STYLE['card'], 'plot_bgcolor': STYLE['card'],
        'margin': dict(t=35, b=30, l=40, r=20), 'font': {'size': 10, 'color': STYLE['text']},
        'title': {'font': {'size': 12, 'color': STYLE['accent']}, 'y': 0.95}
    }

    # Gráficos ajustados
    f1 = px.histogram(dff, x="Gender", color="Subscription Status", barmode="group",
                      color_discrete_sequence=[STYLE['accent'], '#1B4D3E'], title="SUBSCRIPTION_BY_GENDER")
    
    f2 = px.histogram(dff, x="Purchase Amount (USD)", color_discrete_sequence=[STYLE['accent']], 
                      title="PURCHASE_INTENSITY")
    
    ct = pd.crosstab(dff['Discount Applied'], dff['Subscription Status'], normalize='index')
    f3 = px.imshow(ct, text_auto=".2f", color_continuous_scale=[STYLE['bg'], STYLE['accent']], 
                   title="PROBABILITY_MATRIX")
    
    f4 = px.scatter(dff, x="Age", y="Purchase Amount (USD)", color_discrete_sequence=[STYLE['accent']], 
                    opacity=0.3, title="AGE_DISPERSION")

    for f in [f1, f2, f3, f4]: f.update_layout(layout_cfg)
    
    return f1, f2, f3, f4, kpi1, kpi2, kpi3

if __name__ == '__main__':
    app.run(debug=True)