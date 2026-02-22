import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Cargar datos
df = pd.read_csv('shopping_behavior_updated.csv')

# Inicializar la App con un tema profesional (SLATE es oscuro y moderno)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# --- DISEÑO DE COMPONENTES ---

header = html.Div([
    html.H2("Análisis de Comportamiento de Compra", className="display-4 text-center mt-4"),
    html.P("Insights predictivos y segmentación de clientes", className="lead text-center mb-5"),
    html.Hr(style={'borderColor': 'white'})
], className="container")

controls = dbc.Card([
    dbc.CardBody([
        html.H5("Controles de Filtro", className="card-title"),
        html.Label("Categoría de Producto"),
        dcc.Dropdown(
            id='category-filter',
            options=[{'label': c, 'value': c} for c in df['Category'].unique()] + [{'label': 'Todas', 'value': 'all'}],
            value='all',
            className="mb-3",
            style={'color': '#333'}
        ),
        html.Label("Rango de Edad"),
        dcc.RangeSlider(
            id='age-slider',
            min=df['Age'].min(),
            max=df['Age'].max(),
            value=[df['Age'].min(), df['Age'].max()],
            marks={i: str(i) for i in range(20, 71, 10)},
            step=1
        ),
    ])
], className="mb-4 shadow")

# --- LAYOUT PRINCIPAL ---

app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='subscription-plot', className="shadow"), md=6),
                dbc.Col(dcc.Graph(id='amount-dist-plot', className="shadow"), md=6),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='discount-heatmap', className="shadow"), md=6),
                dbc.Col(dcc.Graph(id='age-scatter-plot', className="shadow"), md=6),
            ])
        ], md=8)
    ])
], fluid=True)

# --- LÓGICA (CALLBACKS) ---

@app.callback(
    [Output('subscription-plot', 'figure'),
     Output('amount-dist-plot', 'figure'),
     Output('discount-heatmap', 'figure'),
     Output('age-scatter-plot', 'figure')],
    [Input('category-filter', 'value'),
     Input('age-slider', 'value')]
)
def update_graphs(selected_category, age_range):
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    if selected_category != 'all':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # Paleta de colores consistente
    colors = {'Yes': '#00fa9a', 'No': '#ff4b4b'}

    # 1. Suscripción por Género
    fig1 = px.histogram(filtered_df, x="Gender", color="Subscription Status", 
                        barmode="group", title="Suscripción por Género",
                        template="plotly_dark", color_discrete_map=colors)
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 2. Distribución de Gasto
    fig2 = px.histogram(filtered_df, x="Purchase Amount (USD)", 
                        title="Distribución de Tickets de Venta",
                        template="plotly_dark", color_discrete_sequence=['#4da6ff'])
    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 3. Heatmap Descuento
    ct = pd.crosstab(filtered_df['Discount Applied'], filtered_df['Subscription Status'], normalize='index')
    fig3 = px.imshow(ct, text_auto=".2f", title="Correlación: Descuento vs Suscripción",
                     template="plotly_dark", color_continuous_scale='Blues')
    fig3.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 4. Edad vs Gasto
    fig4 = px.scatter(filtered_df, x="Age", y="Purchase Amount (USD)", color="Category",
                      title="Análisis de Dispersión: Edad vs Gasto",
                      template="plotly_dark", opacity=0.6)
    fig4.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run(debug=True)