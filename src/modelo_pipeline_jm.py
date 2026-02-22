"""
PIPELINE ML — CUSTOMER SHOPPING BEHAVIOR

"""

# ================= LIBRERÍAS =================
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

# ================= REPRODUCIBILIDAD =================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ================= CONFIG =================
DATA_PATH = "data/raw/shopping_behavior_updated.csv"
TARGET = "Purchase Amount (USD)"

# ================= CARGA =================
df = pd.read_csv(DATA_PATH)

# ================= LIMPIEZA =================
df = df.drop_duplicates()
df = df.dropna()

# eliminar IDs
if "Customer ID" in df.columns:
    df = df.drop(columns=["Customer ID"])

# ================= FEATURE ENGINEERING =================

# evitar división por cero
if "Previous Purchases" in df.columns:
    df["avg_spend_per_purchase"] = df[TARGET] / (df["Previous Purchases"] + 1)

# segmentación edad
if "Age" in df.columns:
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0,25,40,60,100],
        labels=["young","adult","mid","senior"]
    )

# cliente frecuente
if "Previous Purchases" in df.columns:
    df["is_loyal"] = (df["Previous Purchases"] > 10).astype(int)

# ================= CORRELACIONES =================
print("\n===== CORRELACIÓN CON TARGET =====")
corr = df.corr(numeric_only=True)[TARGET].sort_values(ascending=False)
print(corr)

# ================= SPLIT =================
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# ================= COLUMNAS =================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ================= PREPROCESAMIENTO =================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ================= MODELOS =================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=SEED),
    "GradientBoost": GradientBoostingRegressor(random_state=SEED)
}

params = {
    "RandomForest": {
        "model__n_estimators":[200,400],
        "model__max_depth":[5,10,None],
        "model__min_samples_split":[2,5]
    },
    "GradientBoost":{
        "model__learning_rate":[0.01,0.05,0.1],
        "model__n_estimators":[200,400],
        "model__max_depth":[2,3,4]
    }
}

best_model = None
best_score = np.inf

print("\n===== ENTRENANDO MODELOS =====")

for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    if name in params:
        grid = GridSearchCV(
            pipe,
            params[name],
            cv=10,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        fitted = grid.best_estimator_
    else:
        fitted = pipe.fit(X_train, y_train)

    preds = fitted.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)

    if rmse < best_score:
        best_score = rmse
        best_model = fitted
        best_name = name

print("\n🏆 MEJOR MODELO:", best_name)

# ================= IMPORTANCIA =================
def mostrar_importancia(model, nombre):
    try:
        importances = model.named_steps["model"].feature_importances_
        feature_names = model.named_steps["prep"].get_feature_names_out()

        imp = pd.Series(importances, index=feature_names)\
            .sort_values(ascending=False)

        print(f"\nTOP VARIABLES — {nombre}")
        print(imp.head(15))

    except:
        print(f"\n{nombre} no soporta feature importance")

mostrar_importancia(best_model, best_name)