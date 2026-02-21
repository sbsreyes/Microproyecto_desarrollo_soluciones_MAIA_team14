# Guía: Ejecutar MLflow en EC2 (AWS)

## 1. Conexión a EC2
Desde tu computador (Mac):

chmod 400 maia-microproyecto-ec2.pem
ssh -i maia-microproyecto-ec2.pem ubuntu@IP_PUBLICA

## 2. Crear entorno virtual en EC2
sudo apt update
sudo apt install python3-venv -y

python3 -m venv venv
source venv/bin/activate

## 3. Instalar dependencias
pip install -r requirements.txt
pip install mlflow

Verificar:
mlflow --version

## 4. Levantar MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

## 5. Abrir puerto 5000
En AWS:
EC2 → Security Groups → Inbound rules
Agregar:
Custom TCP | Port 5000 | Source 0.0.0.0/0

## 6. Acceder desde navegador
http://IP_PUBLICA:5000
