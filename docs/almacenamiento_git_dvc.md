# Almacenamiento de código y datos

Como parte del desarrollo del micro-proyecto, se estableció un esquema de almacenamiento y versionamiento de código y datos que permite garantizar la trazabilidad de los cambios y la reproducibilidad del análisis. Para el control de versiones del código y de los artefactos livianos se utilizó Git, mientras que para el manejo de los conjuntos de datos se empleó DVC, separando los archivos de gran tamaño del repositorio de código.

El repositorio Git almacena el código fuente del proyecto, los scripts utilizados para el análisis exploratorio de datos, las dependencias del entorno y los artefactos livianos generados durante el EDA, tales como tablas resumen y figuras. Este enfoque facilita el trabajo colaborativo y el seguimiento de los aportes realizados por los integrantes del equipo.

Por su parte, DVC se utilizó para versionar los datos en bruto y los datos procesados, permitiendo mantener un historial de las versiones del dataset sin necesidad de almacenar los archivos directamente en Git. En particular, se versionaron el conjunto de datos original y el dataset limpio generado durante la etapa de exploración. Los archivos `.dvc` correspondientes se almacenan en el repositorio, mientras que los datos reales se gestionan a través del remote configurado en DVC.

Este esquema de almacenamiento permite que cualquier integrante del equipo pueda reproducir el análisis realizado, obteniendo tanto el código como los datos necesarios mediante los comandos de Git y DVC, y constituye la base para las etapas posteriores de desarrollo del proyecto.

