"""
Micro-proyecto Datos - Entrega 2
Pipeline de modelamiento para predicción de Subscription Status y Purchase Amount
Dataset: Shopping Behavior (Kaggle)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN MLFLOW
# ============================================================
# Apuntar al servidor MLflow en EC2:
# mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")
mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "shopping_behavior_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# ============================================================
# 1. CONSTANTES DE PREPROCESAMIENTO
# ============================================================
FREQ_MAP = {
    'Weekly': 52, 'Fortnightly': 26, 'Bi-Weekly': 26,
    'Monthly': 12, 'Every 3 Months': 4, 'Quarterly': 4, 'Annually': 1
}

NORTHEAST = ['Maine','New Hampshire','Vermont','Massachusetts','Rhode Island',
             'Connecticut','New York','New Jersey','Pennsylvania']
SOUTHEAST = ['Delaware','Maryland','Virginia','West Virginia','North Carolina',
             'South Carolina','Georgia','Florida','Kentucky','Tennessee',
             'Alabama','Mississippi','Arkansas','Louisiana']
MIDWEST   = ['Ohio','Indiana','Illinois','Michigan','Wisconsin','Minnesota',
             'Iowa','Missouri','North Dakota','South Dakota','Nebraska','Kansas']
SOUTHWEST = ['Texas','Oklahoma','New Mexico','Arizona']

WARM_COLORS = ['Red','Orange','Yellow','Maroon','Pink','Peach','Gold',
               'Magenta','Salmon','Terra cotta','Burgundy','Brown']
COOL_COLORS  = ['Blue','Green','Purple','Teal','Turquoise','Cyan',
                'Indigo','Violet','Lavender','Olive']

def get_region(state):
    if state in NORTHEAST: return 'Northeast'
    if state in SOUTHEAST: return 'Southeast'
    if state in MIDWEST:   return 'Midwest'
    if state in SOUTHWEST: return 'Southwest'
    return 'West'

def get_color_group(color):
    if color in WARM_COLORS: return 'Warm'
    if color in COOL_COLORS: return 'Cool'
    return 'Neutral'

# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 1. Eliminaciones y limpieza inicial
    # Mantenemos 'Discount Applied' momentáneamente para crear una feature
    d['Has_Discount'] = (d['Discount Applied'] == 'Yes').astype(int)
    d = d.drop(columns=['Customer ID', 'Item Purchased', 'Discount Applied'])

    # 2. Encodings binarios y mapeo de frecuencia
    d['Gender_Is_Male']     = (d['Gender'] == 'Male').astype(int)
    d['Subscription Status']= (d['Subscription Status'] == 'Yes').astype(int)
    d['Promo Code Used']    = (d['Promo Code Used'] == 'Yes').astype(int)
    d['Frequency_Numeric']  = d['Frequency of Purchases'].map(FREQ_MAP)

    # 3. Ingeniería de Variables (Para combatir el R2 negativo)
    # Creamos interacciones que ayuden al modelo a ver patrones de gasto
    d['Age_x_Freq'] = d['Age'] * d['Frequency_Numeric']
    d['Total_Engagement'] = d['Previous Purchases'] * d['Review Rating']

    # 4. Reducción de cardinalidad
    d['Region']      = d['Location'].map(get_region)
    d['Color Group'] = d['Color'].map(get_color_group)
    d = d.drop(columns=['Location', 'Color', 'Gender', 'Frequency of Purchases'])

    # 5. One-Hot Encoding (Usamos drop_first=True para evitar redundancia)
    nominal_cols = ['Category', 'Season', 'Size', 'Shipping Type',
                    'Payment Method', 'Region', 'Color Group']
    d = pd.get_dummies(d, columns=nominal_cols, drop_first=True, dtype=int)

    return d


# ============================================================
# 3. CARGA DE DATOS Y SPLITS
# ============================================================
def load_and_split(data_path: str):
    df_raw = pd.read_csv(data_path)
    d = preprocess(df_raw)

    # --- Modelo A: Clasificación — Subscription Status ---
    # Se elimina Promo Code Used porque tiene correlación perfecta con el target
    # (todos los suscriptores usan promo codes). Incluirlo sería data leakage.
    X_clf = d.drop(columns=['Subscription Status', 'Promo Code Used']).values.astype(float)
    y_clf = d['Subscription Status'].values
    clf_features = d.drop(columns=['Subscription Status', 'Promo Code Used']).columns.tolist()

    # --- Modelo B: Regresión — Purchase Amount ---
    # Subscription Status se mantiene como feature (es información legítima en
    # el contexto de predecir cuánto gasta un cliente)
    X_reg = d.drop(columns=['Purchase Amount (USD)']).values.astype(float)
    y_reg = d['Purchase Amount (USD)'].values.astype(float)
    reg_features = d.drop(columns=['Purchase Amount (USD)']).columns.tolist()

    # Splits estratificados (clasificación) y aleatorios (regresión)
    splits_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    splits_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    return splits_clf, splits_reg, clf_features, reg_features


# ============================================================
# 4. ENTRENAMIENTO CON MLFLOW
# ============================================================
def train_classifier(X_train, X_test, y_train, y_test, model, model_name, scaler=None):
    """Entrena un clasificador y registra métricas en MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Scaling si aplica
        if scaler:
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_prob  = model.predict_proba(X_te)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_prob),
        }

        # Log params
        mlflow.log_params(model.get_params())
        # Log metrics
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n  [{model_name}]")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print(classification_report(y_test, y_pred, target_names=['No Subscribed', 'Subscribed']))

        return model, metrics


def train_regressor(X_train, X_test, y_train, y_test, model, model_name, scaler=None):
    """Entrena un regresor y registra métricas en MLflow."""
    with mlflow.start_run(run_name=model_name):
        if scaler:
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae":  float(mean_absolute_error(y_test, y_pred)),
            "r2":   float(r2_score(y_test, y_pred)),
        }

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n  [{model_name}]")
        for k, v in metrics.items():
            print(f"  {k:6s}: {v:.4f}")

        return model, metrics


# ============================================================
# 5. EJECUCIÓN PRINCIPAL
# ============================================================
if __name__ == "__main__":
    DATA_PATH = "data/raw/shopping_behavior_updated.csv"

    print("Cargando y preprocesando datos...")
    (Xct, Xce, yct, yce), (Xrt, Xre, yrt, yre), clf_cols, reg_cols = load_and_split(DATA_PATH)

    sc_clf = StandardScaler()
    sc_reg = StandardScaler()

    # ---- CLASIFICACIÓN ----
    print("\n" + "="*60)
    print("MODELO A — Clasificación: Subscription Status")
    print("="*60)
    print(f"  Train: {Xct.shape[0]} | Test: {Xce.shape[0]}")
    print(f"  Balance train: {yct.mean():.2%} suscriptores")

    train_classifier(Xct, Xce, yct, yce,
                     LogisticRegression(max_iter=1000, random_state=42),
                     "LogisticRegression_Subscription", scaler=StandardScaler())

    rfc, _ = train_classifier(Xct, Xce, yct, yce,
                              RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
                              "RandomForest_Subscription", scaler=None)

    # Feature importance
    print("\n  Top 10 features (RF Clasificación):")
    fi = pd.Series(rfc.feature_importances_, index=clf_cols).sort_values(ascending=False)
    print(fi.head(10).to_string())

    # ---- REGRESIÓN ----
    print("\n" + "="*60)
    print("MODELO B — Regresión: Purchase Amount (USD)")
    print("="*60)
    print(f"  Train: {Xrt.shape[0]} | Test: {Xre.shape[0]}")
    print(f"  Target mean: ${yrt.mean():.2f} | std: ${yrt.std():.2f}")

    train_regressor(Xrt, Xre, yrt, yre,
                    LinearRegression(),
                    "LinearRegression_PurchaseAmount", scaler=StandardScaler())

    rfr, _ = train_regressor(Xrt, Xre, yrt, yre,
                              RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                              "RandomForest_PurchaseAmount", scaler=None)

    print("\n  Top 10 features (RF Regresión):")
    fi2 = pd.Series(rfr.feature_importances_, index=reg_cols).sort_values(ascending=False)
    print(fi2.head(10).to_string())

    print("\n✓ Experimentos registrados en MLflow.")