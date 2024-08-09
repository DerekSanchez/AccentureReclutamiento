
from sklearn.linear_model import LogisticRegression  #Regresión logística
from sklearn.svm import SVC  #Support vector machine
from sklearn.neighbors import KNeighborsClassifier  #k-NN
from sklearn.tree import DecisionTreeClassifier  #Árboles de decisión
from sklearn.ensemble import RandomForestClassifier  #random forest
from sklearn.ensemble import GradientBoostingClassifier  #Gradient Boosting
from sklearn.naive_bayes import GaussianNB  #Naive Bayes Gaussiano

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


#Funcion que permite que el usuario seleccione un modelo y una metrica de evaluacion a maximizar
#Toma de input los sets de entrenamiento y testing
#Arroja de output los mejores parámetros del modelo (basados en randomized grid search) y las metricas relevantes para un problema de clasificacion
def SeleccionModelo(X_train, y_train, X_test, y_test, ModelSelected, metric, seed, iteraciones):

    #Lista de modelos disponibles
    modelos_disponibles = [
        "Logistic Regression", "SVM", "KNN", 
        "Decision Tree", "Random Forest", 
        "Gradient Boosting"
    ]

    #Codicion if para setear el modelo y el grid de parametros dependiendo de la seleccion del usuario
    if ModelSelected == "Logistic Regression":
        modelo = LogisticRegression()
        parametros = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500]
        }
        
    elif ModelSelected == "SVM":
        modelo = SVC()
        parametros = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1, 10],
            'degree': [3, 4, 5]  
        }
        
    elif ModelSelected == "KNN":
        modelo = KNeighborsClassifier()
        parametros = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2] 
        }
        
    elif ModelSelected == "Decision Tree":
        modelo = DecisionTreeClassifier()
        parametros = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }
        
    elif ModelSelected == "Random Forest":
        modelo = RandomForestClassifier()
        parametros = {
            'n_estimators': [100, 200, 300, 500],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
    elif ModelSelected == "Gradient Boosting":
        modelo = GradientBoostingClassifier()
        parametros = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }

    else:
        #Advertencia para el usuario
        print("Advertencia: El modelo seleccionado no es válido.")
        print(f"Escoge uno de los modelos disponibles: {', '.join(modelos_disponibles)}")
        modelo = None
        parametros = None
    
    scoring_options = [
    'accuracy',          #Accuracy
    'f1',                #F1-score (media armónica de precisión y recall)
    'precision',         #Precisión (proporción de verdaderos positivos sobre el total de positivos predichos)
    'recall',            #Recall o sensibilidad (proporción de verdaderos positivos sobre el total de verdaderos positivos y falsos negativos)
    'roc_auc',           #AUC de la curva ROC (útil para clasificación binaria)
    'average_precision', #Precisión promedio ponderada (similar a AUC pero para precisión y recall)
    'log_loss',          #Logarithmic Loss (medida de incertidumbre de las predicciones)
    'balanced_accuracy', #accuraccy balanceada (promedio de recall por clase)
    ]
    
    #asignación de la métrica de scoring
    if metric not in scoring_options:
        print("Advertencia: La métrica de scoring ingresada no es válida.")
        print("Escoge una de las métricas disponibles: accuracy, f1, precision, recall, roc_auc, average_precision, log_loss o balanced_accuracy.")
        scoring = None

    #Iniciar grid
    random_search = RandomizedSearchCV(modelo, param_distributions = parametros, cv=5, scoring= metric, random_state= seed, n_iter = iteraciones)

    #Fitting de grid
    random_search.fit(X_train, y_train)

    #Extrae el mejor modelo por grid
    best_model = random_search.best_estimator_

    #Predice en el test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    #Evalua el modelo
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    #imprime las metricas de performance
    print(f'Mejores Parametros: {random_search.best_params_}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    #Matriz de confusion
    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    return best_model