import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# 1. Cargar y limpiar datos
df = pd.read_csv('data/raw/shopping_behavior_updated.csv')

# Inicializar la App
app = dash.Dash(__name__)

# 2. Diseño de la Interfaz (Layout)
app.layout = html.Div(style={'backgroundColor': '#1e1e1e', 'color': 'white', 'padding': '20px', 'fontFamily': 'Arial'}, children=[
    html.H1("Dashboard de Análisis: Comportamiento de Compra", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.Label("Seleccionar Categoría:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': c, 'value': c} for c in df['Category'].unique()] + [{'label': 'Todas', 'value': 'all'}],
                value='all',
                style={'color': 'black'}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Rango de Edad:"),
            dcc.RangeSlider(
                id='age-slider',
                min=df['Age'].min(),
                max=df['Age'].max(),
                value=[df['Age'].min(), df['Age'].max()],
                marks={i: str(i) for i in range(15, 75, 5)}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'backgroundColor': '#2d2d2d', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    html.Div([
        dcc.Graph(id='subscription-plot', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='amount-dist-plot', style={'width': '48%', 'display': 'inline-block'}),
    ]),

    html.Div([
        dcc.Graph(id='discount-heatmap', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='age-scatter-plot', style={'width': '48%', 'display': 'inline-block'}),
    ])
])

# 3. Lógica de Interactividad (Callbacks)
@app.callback(
    [Output('subscription-plot', 'figure'),
     Output('amount-dist-plot', 'figure'),
     Output('discount-heatmap', 'figure'),
     Output('age-scatter-plot', 'figure')],
    [Input('category-filter', 'value'),
     Input('age-slider', 'value')]
)
def update_graphs(selected_category, age_range):
    # Filtrar datos
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    if selected_category != 'all':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # Gráfico 1: Suscripción por Género
    fig1 = px.histogram(filtered_df, x="Gender", color="Subscription Status", 
                        barmode="group", title="Suscripción por Género",
                        template="plotly_dark", color_discrete_map={'Yes': '#00CC96', 'No': '#EF553B'})

    # Gráfico 2: Distribución de Gasto
    fig2 = px.histogram(filtered_df, x="Purchase Amount (USD)", nbins=20,
                        title="Distribución de Montos de Compra",
                        template="plotly_dark", color_discrete_sequence=['#636EFA'])

    # Gráfico 3: Matriz Descuento vs Suscripción
    ct = pd.crosstab(filtered_df['Discount Applied'], filtered_df['Subscription Status'], normalize='index')
    fig3 = px.imshow(ct, text_auto=True, title="Probabilidad de Suscripción según Descuento",
                     labels=dict(x="Suscrito", y="Aplicó Descuento", color="Probabilidad"),
                     template="plotly_dark", color_continuous_scale='Viridis')

    # Gráfico 4: Edad vs Gasto (Scatter)
    fig4 = px.scatter(filtered_df, x="Age", y="Purchase Amount (USD)", color="Season",
                      title="Edad vs Gasto por Temporada",
                      template="plotly_dark", opacity=0.7)

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run(debug=True)