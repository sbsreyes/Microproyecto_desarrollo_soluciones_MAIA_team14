"""
EDA - Customer Shopping Behavior

Exploración descriptiva de los datos (EDA) con el objetivo de:
- Evaluar la calidad del dataset
- Identificar patrones generales de compra
- Generar tablas y visualizaciones que apoyen análisis posteriores
- Dejar el dataset limpio y listo para aplicar cualquier tipo de modelo

Salidas principales:
- Tablas descriptivas en reports/tables
- Figuras exploratorias en reports/figures
- Dataset procesado en data/processed
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------
# Definición rutas
#---------------------------------------------------------

RAW_PATH = Path("data/raw/shopping_behavior_updated.csv")
OUT_FIG = Path("reports/figures")
OUT_TAB = Path("reports/tables")
OUT_PROC = Path("data/processed")

#---------------------------------------------------------
# Funciones auxiliares
#---------------------------------------------------------
def snake_case(s: str) -> str:
    """
    Funcion la cual convierte nombres de columnas a 
    formato snake_case.
    - Pasa a minúsculas
    - Elimina caracteres especiales
    - Reemplaza espacios y guiones por guiones bajos
    """    
    s = s.strip().lower()
    for ch in ["(", ")", "%"]:
        s = s.replace(ch, "")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    return s

#---------------------------------------------------------
# Función principal
#---------------------------------------------------------
def main():
    """
    Función que ejecuta el flujo completo
    """    
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_PROC.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # 1) Esquema / faltantes / únicos
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isna().sum().values,
        "n_unique": df.nunique().values
    })
    summary.to_csv(OUT_TAB / "01_schema_missing_unique.csv", index=False)

    # 2) Duplicados
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["Customer ID"].duplicated().sum()) if "Customer ID" in df.columns else np.nan
    pd.DataFrame([{"duplicated_rows": dup_rows, "duplicated_customer_id": dup_ids}]).to_csv(
        OUT_TAB / "02_duplicates.csv", index=False
    )

    # 3) Normalizar nombres columnas
    df2 = df.copy()
    df2.columns = [snake_case(c) for c in df2.columns]

    # 4) Estadísticos numéricos
    num_cols = df2.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if num_cols:
        df2[num_cols].describe().T.to_csv(OUT_TAB / "03_numeric_describe.csv")

    # 5) Top categorías
    cat_cols = df2.select_dtypes(include=["object", "bool"]).columns.tolist()
    for c in cat_cols:
        (df2[c].value_counts(dropna=False).head(20)
            .to_frame("count")
            .to_csv(OUT_TAB / f"04_top_{c}.csv"))

    # 6) Figuras clave
    if "purchase_amount_usd" in df2.columns:
        plt.figure()
        df2["purchase_amount_usd"].plot(kind="hist", bins=30)
        plt.title("Distribución: purchase_amount_usd")
        plt.xlabel("USD"); plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(OUT_FIG / "hist_purchase_amount_usd.png", dpi=200)
        plt.close()

    if "category" in df2.columns and "purchase_amount_usd" in df2.columns:
        plt.figure()
        df2.boxplot(column="purchase_amount_usd", by="category", rot=45)
        plt.title("Purchase amount por category")
        plt.suptitle("")
        plt.xlabel("category"); plt.ylabel("USD")
        plt.tight_layout()
        plt.savefig(OUT_FIG / "box_purchase_by_category.png", dpi=200)
        plt.close()

    if "age" in df2.columns and "purchase_amount_usd" in df2.columns:
        plt.figure()
        plt.scatter(df2["age"], df2["purchase_amount_usd"], s=10)
        plt.title("age vs purchase_amount_usd")
        plt.xlabel("age"); plt.ylabel("purchase_amount_usd")
        plt.tight_layout()
        plt.savefig(OUT_FIG / "scatter_age_vs_purchase.png", dpi=200)
        plt.close()

    # 7) Dataset procesado
    df2.to_csv(OUT_PROC / "shopping_clean.csv", index=False)

    print("EDA lista:")
    print("- reports/tables")
    print("- reports/figures")
    print("- data/processed/shopping_clean.csv")

if __name__ == "__main__":
    main()
