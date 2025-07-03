import pandas as pd
import joblib
import sys
import json

def predict_single_instance(instance, model, model_columns, scaler):
    """Predice si una única instancia de tráfico de red es un ataque."""
    df_instance = pd.DataFrame([instance])
    
    # Identificar columnas numéricas originales (excluyendo las que se crearán con one-hot)
    numerical_cols = [col for col in df_instance.columns if df_instance[col].dtype in ['int64', 'float64']]
    
    # Preprocesamiento: One-Hot Encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        if col in df_instance.columns:
            df_instance[col] = df_instance[col].astype('category')
    df_encoded = pd.get_dummies(df_instance, columns=categorical_cols, dummy_na=False)
    
    # Alinear columnas con las del modelo
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Preprocesamiento: Escalado de Características
    # Usar el scaler cargado para transformar las columnas numéricas
    if numerical_cols: # Asegurarse de que hay columnas numéricas para escalar
        df_aligned[numerical_cols] = scaler.transform(df_aligned[numerical_cols])
    
    # Realizar la predicción
    prediction = model.predict(df_aligned)[0]
    probability = model.predict_proba(df_aligned)[0]
    
    return prediction, probability

def main():
    """Función principal para ejecutar la predicción desde la línea de comandos."""
    if len(sys.argv) != 2:
        print("Uso: python predict.py '<JSON_de_la_instancia>'")
        sys.exit(1)
        
    try:
        instance_json = sys.argv[1]
        instance = json.loads(instance_json)
    except json.JSONDecodeError:
        print("Error: El argumento no es un JSON válido.")
        sys.exit(1)

    # Cargar el modelo, columnas y escalador
    try:
        model = joblib.load('ids_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("Error: No se encontraron los archivos del modelo o el escalador.")
        print("Por favor, ejecuta 'train_model.py' primero.")
        sys.exit(1)
        
    # Realizar la predicción
    prediction, probability = predict_single_instance(instance, model, model_columns, scaler)
    
    print("\n--- Resultado de la Detección (con Escalado) ---")
    if prediction == 1:
        print("Resultado: ¡ALERTA! Se ha detectado un posible ATAQUE.")
        print(f"Probabilidad de ser un ataque: {probability[1]:.2%}")
    else:
        print("Resultado: El tráfico parece ser NORMAL.")
        print(f"Probabilidad de ser normal: {probability[0]:.2%}")
    print("---------------------------------------------")

if __name__ == "__main__":
    main()