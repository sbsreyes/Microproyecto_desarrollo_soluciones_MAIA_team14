"""
PIPELINE ML — CUSTOMER SHOPPING BEHAVIOR

Pipeline completo para:
- Predicción de monto de compra (regresión)

"""

# Librerías
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)

# Configuración
DATA_PATH = "data/raw/shopping_behavior_updated.csv"
TARGET_REG = "Purchase Amount (USD)"
TARGET_CLF = "Subscription Status"

# Carga
df = pd.read_csv(DATA_PATH)

# Limpieza
df = df.drop_duplicates()
df = df.dropna()

# Convertir target clasificación a binario
df[TARGET_CLF] = df[TARGET_CLF].map({"Yes":1,"No":0})

# Particionamiento
X = df.drop(columns=[TARGET_REG, TARGET_CLF])
y_reg = df[TARGET_REG]
y_clf = df[TARGET_CLF]

# Columnas
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Pre procesor
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Modelo
X_train, X_test, yreg_train, yreg_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

_, _, yclf_train, yclf_test = train_test_split(
    X, y_clf, test_size=0.3, random_state=42
)

# ================= Modelos de regresión =====================
reg_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoost": GradientBoostingRegressor()
}

reg_params = {
    "RandomForest": {
        "model__n_estimators":[100,200],
        "model__max_depth":[5,10,None]
    },
    "GradientBoost":{
        "model__learning_rate":[0.01,0.1],
        "model__n_estimators":[100,200]
    }
}

best_reg = None
best_score = np.inf

print("\n===== ENTRENANDO MODELOS REGRESIÓN =====")

for name, model in reg_models.items():

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    if name in reg_params:
        grid = GridSearchCV(pipe, reg_params[name],
                            cv=5, scoring="neg_root_mean_squared_error")
        grid.fit(X_train, yreg_train)
        best_model = grid.best_estimator_
    else:
        pipe.fit(X_train, yreg_train)
        best_model = pipe

    preds = best_model.predict(X_test)
    rmse = mean_squared_error(yreg_test, preds, squared=False)
    r2 = r2_score(yreg_test, preds)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)

    if rmse < best_score:
        best_score = rmse
        best_reg = best_model
        best_reg_name = name

print("\n🏆 Mejor modelo regresión:", best_reg_name)

# ================= IMPORTANCIA VARIABLES =================

def mostrar_importancia(model, nombre):
    try:
        importances = model.named_steps["model"].feature_importances_
        feature_names = model.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print(f"\nTOP VARIABLES — {nombre}")
        print(imp.head(10))
    except:
        print(f"\n{nombre} no soporta feature importance")

mostrar_importancia(best_reg, best_reg_name)