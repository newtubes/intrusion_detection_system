import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ... (COLUMNS y load_data() se mantienen igual)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_data():
    """Carga los datasets de entrenamiento y prueba de NSL-KDD."""
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'KDDTrain+.txt')
    test_path = os.path.join(os.path.dirname(__file__), 'data', 'KDDTest+.txt')
    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=COLUMNS)
    return df_train, df_test

def preprocess_data(df_train, df_test):
    # ... (La función de preprocesamiento se mantiene exactamente igual)
    print("Preprocesando datos...")
    df_train['is_attack'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_test['is_attack'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_train.drop(columns=['label', 'difficulty'], inplace=True)
    df_test.drop(columns=['label', 'difficulty'], inplace=True)
    numerical_cols = df_train.select_dtypes(include=['number']).columns.tolist()
    numerical_cols.remove('is_attack')
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, dummy_na=False)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, dummy_na=False)
    train_labels = df_train_encoded['is_attack']
    test_labels = df_test_encoded['is_attack']
    train_features = df_train_encoded.drop(columns=['is_attack'])
    test_features = df_test_encoded.drop(columns=['is_attack'])
    train_features, test_features = train_features.align(test_features, join='inner', axis=1, fill_value=0)
    scaler = StandardScaler()
    train_features[numerical_cols] = scaler.fit_transform(train_features[numerical_cols])
    test_features[numerical_cols] = scaler.transform(test_features[numerical_cols])
    print("Preprocesamiento completado.")
    return train_features, train_labels, test_features, test_labels, scaler

def main():
    """Función principal para entrenar y evaluar el modelo con ajuste de hiperparámetros."""
    df_train, df_test = load_data()
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df_train, df_test)
    
    # Definir la parrilla de hiperparámetros para la búsqueda
    # NOTA: Esta es una parrilla pequeña para que la ejecución sea rápida. 
    # En un caso real, probaríamos más combinaciones.
    param_grid = {
        'n_estimators': [100, 150],       # Número de árboles
        'max_depth': [20, 30],            # Profundidad máxima de los árboles
        'min_samples_leaf': [1, 2]        # Mínimo de muestras por hoja
    }
    
    # Configurar la búsqueda con validación cruzada (Grid Search)
    # cv=3 significa 3-fold cross-validation. n_jobs=-1 usa todos los cores de la CPU.
    # verbose=2 muestra el progreso del entrenamiento.
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
                               param_grid=param_grid, 
                               cv=3, 
                               verbose=2, 
                               n_jobs=-1)
    
    print("\nIniciando el ajuste de hiperparámetros (Grid Search)... Esto puede tardar.")
    grid_search.fit(X_train, y_train)
    
    print("\nAjuste de hiperparámetros completado.")
    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    
    # Obtener el mejor modelo encontrado
    best_model = grid_search.best_estimator_
    
    # Evaluar el mejor modelo
    print("\nEvaluando el modelo optimizado con el conjunto de prueba...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión (Accuracy) del modelo optimizado: {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Ataque']))
    
    # Guardar el modelo optimizado y los otros artefactos
    joblib.dump(best_model, 'ids_model.pkl')
    joblib.dump(X_train.columns, 'model_columns.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nModelo optimizado, columnas y escalador guardados exitosamente.")

if __name__ == "__main__":
    main()