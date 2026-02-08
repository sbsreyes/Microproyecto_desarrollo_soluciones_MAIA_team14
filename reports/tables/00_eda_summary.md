# Resumen EDA – shopping_behavior_updated.csv

## Artefactos generados
- `reports/tables/01_schema_missing_unique.csv`: tipos de datos, faltantes, cardinalidad por columna.
- `reports/tables/02_duplicates.csv`: filas duplicadas y duplicados por `Customer ID` (si aplica).
- `reports/tables/03_numeric_describe.csv`: estadísticos descriptivos de variables numéricas.
- `reports/tables/04_top_*.csv`: top 20 categorías por variable categórica.
- `reports/figures/`: histogramas y gráficos exploratorios.
- `data/processed/shopping_clean.csv`: versión “limpia” con nombres de columnas normalizados.

## Hallazgos (plantilla)
1) Calidad de datos: revisar faltantes y duplicados (ver `01_schema...` y `02_duplicates...`).
2) Distribución de gasto: revisar dispersión/cola (ver `hist_purchase_amount_usd.png`).
3) Variación por categoría: diferencias en gasto entre categorías (ver `box_purchase_by_category.png`).
4) Relación edad–gasto: tendencia visual (ver `scatter_age_vs_purchase.png`).

## Reproducibilidad
Ejecutar:
- `python src/eda_shopping.py`
para regenerar tablas/figuras y el dataset procesado.
