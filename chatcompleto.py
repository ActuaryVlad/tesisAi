import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, recall_score, f1_score



train = pd.read_csv('C:/Users/monte/Desktop/tesis/train.csv')
train = pd.read_csv('C:/Users/monte/Desktop/tesis/train.csv')
test = pd.read_csv('C:/Users/monte/Desktop/tesis/test.csv')

print("Datos de entrenamiento cuerpo:", train.shape) 
print("Datos de prueba cuerpo:", test.shape)   

train.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)
test.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)


datosnumericos=['Age','Annual_Premium','Vintage']
print("Estadisticas descriptivas de los datos numericos:")
train[datosnumericos].describe()

Variables_categoricas=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

nor = ['Age', 'Vintage']
standard_scaler = StandardScaler().fit(train[nor])
train[nor] = standard_scaler.transform(train[nor])
test[nor] = standard_scaler.transform(test[nor])  # Utilizar el mismo escalador

mm = MinMaxScaler().fit(train[['Annual_Premium']])
train[['Annual_Premium']] = mm.transform(train[['Annual_Premium']])
test[['Annual_Premium']] = mm.transform(test[['Annual_Premium']])  # Utilizar el mismo escalador

# Separar las etiquetas
train_target = train['Response']
test_target = test['Response']
train = train.drop(['Response'], axis=1)
test = test.drop(['Response'], axis=1)

# División en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=0)

# Parámetros para la búsqueda
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 10],
    'min_samples_leaf': [4, 6, 8],
    'min_samples_split': [5, 7, 10]
}

# Crea un clasificador de árbol de decisión
tree_clf = DecisionTreeClassifier()

# Configura la búsqueda aleatoria
random_search = RandomizedSearchCV(tree_clf, param_distributions=param_dist, n_iter=10, cv=4, verbose=1, random_state=42, n_jobs=-1)

# Ajusta el modelo
random_search.fit(x_train, y_train)
# 3. Evaluar el Modelo
print("Mejores parámetros: ", random_search.best_params_)
best_model = random_search.best_estimator_
test_accuracy = best_model.score(x_test, y_test)
print("Precisión en el conjunto de prueba: ", test_accuracy)

#Guardar el arbol en pdf 
dot_data = export_graphviz(best_model, out_file=None, feature_names=train.columns,
                           class_names=['No', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Decision_Tree")  # Esto guardará el árbol en un archivo llamado "Decision_Tree"

dot_data = export_graphviz(best_model, max_depth = 3,
                      out_file=None, 
                      feature_names=train.columns,       # Cambiado de X.columns a train.columns
                      class_names=['Yes','No'],
                      filled=True, rounded=True,
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

# Obtener las probabilidades de la clase positiva
y_probs = best_model.predict_proba(x_test)[:, 1]

# Calcular el AUC-ROC
auc_roc = roc_auc_score(y_test, y_probs)
print("AUC-ROC:", auc_roc)

# Predicciones en el conjunto de prueba
y_pred = best_model.predict(x_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)