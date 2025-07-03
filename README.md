# Sistema de Detección de Intrusiones (IDS) con Machine Learning by Rebeca Romcy

Este proyecto implementa un Sistema de Detección de Intrusiones (IDS) de alto rendimiento, capaz de clasificar tráfico de red como **normal** o **ataque**. Utiliza un modelo de **Random Forest** optimizado mediante búsqueda de hiperparámetros, entrenado con el dataset de referencia **NSL-KDD**.

Este repositorio es una demostración de un pipeline de Machine Learning de extremo a extremo, desde el preprocesamiento de datos complejos hasta la optimización y despliegue de un modelo predictivo.

## Características Principales

- **Modelo Optimizado:** Utiliza un `RandomForestClassifier` cuyos hiperparámetros han sido afinados mediante `GridSearchCV` para maximizar el rendimiento.
- **Dataset Estándar de la Industria:** Entrenado y evaluado con el dataset **NSL-KDD**, una elección común y respetada para la investigación en sistemas de detección de intrusiones.
- **Pipeline de Preprocesamiento Avanzado:**
    - **One-Hot Encoding:** Transforma limpiamente características categóricas (protocolo, servicio, flag) en un formato numérico.
    - **Escalado de Características:** Normaliza las características numéricas con `StandardScaler` para asegurar que el modelo funcione de manera robusta y eficiente.
- **Persistencia Completa:** Guarda y carga no solo el modelo, sino también el escalador y el esquema de columnas, asegurando que las predicciones sean consistentes y reproducibles.

## Tecnologías Utilizadas

*   Python 3
*   Pandas
*   Scikit-learn
*   Joblib

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone (https://github.com/newtubes/intrusion_detection_system)
    cd intrusion_detection_system
    ```

2.  **(Recomendado) Crea y activa un entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Uso del Proyecto

### Paso 1: Entrenamiento y Optimización del Modelo

El script `train_model.py` es el corazón del proyecto. Automatiza todo el proceso de entrenamiento y optimización:

1.  Carga y preprocesa los datos de entrenamiento y prueba del NSL-KDD.
2.  **Ejecuta una búsqueda en parrilla (`GridSearchCV`)** para encontrar la mejor combinación de hiperparámetros para el modelo Random Forest. Este proceso prueba múltiples configuraciones para asegurar el máximo rendimiento.
3.  Entrena el modelo final con los parámetros óptimos.
4.  Evalúa el rendimiento del modelo optimizado y muestra un reporte detallado.
5.  Guarda los tres artefactos esenciales para la predicción: `ids_model.pkl` (el modelo), `model_columns.pkl` (el esquema de datos) y `scaler.pkl` (el escalador).

Para ejecutar el entrenamiento, simplemente corre:
```bash
# Nota: Este proceso puede tardar varios minutos debido a la búsqueda de hiperparámetros.
python train_model.py
```

### Paso 2: Detección de Ataques en Tiempo Real

Con el modelo ya entrenado y guardado, `predict.py` puede clasificar nuevas conexiones de red. El script espera recibir los datos de la conexión en formato **JSON**.

**Ejemplo de Tráfico NORMAL:**
```bash
python predict.py "{\"duration\": 0, \"protocol_type\": \"tcp\", ...}"
```

**Ejemplo de un ATAQUE (DoS):**
```bash
python predict.py "{\"duration\": 0, \"protocol_type\": \"tcp\", ...}"
```


