import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, recall_score, f1_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('C:/Users/monte/Desktop/tesis/train.csv')
test = pd.read_csv('C:/Users/monte/Desktop/tesis/test.csv')

print("Datos de entrenamiento cuerpo:", train.shape) 
print("Datos de prueba cuerpo:", test.shape)   


train.head()

train.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)
test.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)

train.head()

train.dtypes

train.isnull().sum()


datosnumericos=['Age','Annual_Premium','Vintage']
print("Estadisticas descriptivas de los datos numericos:")
train[datosnumericos].describe()

response_counts = train['Response'].value_counts()
response_proportions = train['Response'].value_counts(normalize=True)
print("Cantidad:\n", response_counts)
print("Proporcion:\n", response_proportions)

sns.countplot(x='Response', data=train)
plt.title('Distribución de Respuestas')
plt.show()


plt.hist(train['Age'], bins=20, alpha=0.5, label='Clientes Totales')
plt.hist(train[train['Response'] == 1]['Age'], bins=20, alpha=0.5, label='Clientes que Compraron la Póliza')
plt.xlabel('Edad')
plt.ylabel('Cantidad de Clientes')
plt.legend(loc='upper right')
plt.title('Histograma de Edades')
plt.show()


#Gráfico de densidad (KDE):
sns.kdeplot(train.Age, label="Clientes totales")
sns.kdeplot(train.Age[train.Response == 1], label="Clientes que compraron la poliza")
plt.legend()
plt.show()


# Gráfico de Caja para comparar las edades de los clientes que compraron la póliza con los que no lo hicieron
sns.boxplot(x='Response', y='Age', data=train)
plt.xticks([0, 1], ['No Compraron la Póliza', 'Compraron la Póliza'])
plt.title('Gráfico de Caja de Edades')
plt.show()



plt.hist(train['Annual_Premium'], bins=20, alpha=0.5, label='Clientes Totales')
plt.hist(train[train['Response'] == 1]['Annual_Premium'], bins=20, alpha=0.5, label='Clientes que Compraron la Póliza')
plt.xlabel('Prima Anual')
plt.ylabel('Cantidad de Clientes')
plt.legend(loc='upper right')
plt.title('Histograma de Primas Anuales')
plt.show()



sns.kdeplot(train.Annual_Premium, label="Clientes totales")
sns.kdeplot(train.Annual_Premium[train.Response == 1], label="Clientes que compraron la poliza")
plt.legend()
plt.show()


sns.boxplot(x='Response', y='Annual_Premium', data=train)
plt.xticks([0, 1], ['No Compraron la Póliza', 'Compraron la Póliza'])
plt.title('Gráfico de Caja de Primas Anuales')
plt.show()


plt.hist(train['Vintage'], bins=20, alpha=0.5, label='Clientes Totales')
plt.hist(train[train['Response'] == 1]['Vintage'], bins=20, alpha=0.5, label='Compraron la Póliza')
plt.xlabel('Antigüedad')
plt.ylabel('Cantidad de Clientes')
plt.legend(loc='upper right')
plt.title('Histograma de Antigüedad (Vintage)')
plt.show()



sns.kdeplot(train.Vintage, label="Clientes totales")
sns.kdeplot(train.Vintage[train.Response == 1], label="Compraron la Póliza")
plt.legend()
plt.show()


sns.boxplot(x='Response', y='Vintage', data=train)
plt.xticks([0, 1], ['Clientes Totales', 'Compraron la Póliza'])
plt.title('Gráfico de Caja de Antigüedad (Vintage)')
plt.show()


correlation_matrix = train[['Age', 'Annual_Premium', 'Vintage']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


sns.countplot(x='Gender', hue='Response', data=train)
plt.xlabel('Género')
plt.ylabel('Cantidad de Clientes')
plt.title('Distribución de Género por Respuesta')
plt.legend(title='Respuesta', labels=['No Compraron la Póliza', 'Compraron la Póliza'])
plt.xticks(ticks=[0, 1], labels=['Mujer', 'Hombre'])
plt.show()


cross_tab_gender = pd.crosstab(train['Response'], train['Gender'])
cross_tab_gender.columns = ['Femenino', 'Masculino'] # Puedes ajustar estos nombres según las categorías de tu variable de género
cross_tab_gender.index = ['No Compraron la Póliza', 'Compraron la Póliza']
print(cross_tab_gender)


sns.countplot(x='Driving_License', hue='Response', data=train)
plt.xlabel('Licencia de Conducir')
plt.ylabel('Cantidad de Clientes')
plt.title('Distribución de Licencia de Conducir por Respuesta')
plt.legend(title='Respuesta', labels=['No Compraron la Póliza', 'Compraron la Póliza'])
plt.xticks(ticks=[0, 1], labels=['Sin Licencia', 'Con Licencia'])
plt.show()


cross_tab = pd.crosstab(train['Response'], train['Driving_License'])
cross_tab.columns = ['Sin Licencia', 'Con Licencia']
cross_tab.index = ['No Compraron la Póliza', 'Compraron la Póliza']
print(cross_tab)


sns.countplot(x='Previously_Insured', hue='Response', data=train)
plt.xlabel('Asegurado Previamente')
plt.ylabel('Cantidad de Clientes')
plt.title('Distribución de Aseguramiento Previo por Respuesta')
plt.legend(title='Respuesta', labels=['No Compraron la Póliza', 'Compraron la Póliza'])
plt.xticks(ticks=[0, 1], labels=['No', 'Sí'])
plt.show()


cross_tab_previously_insured = pd.crosstab(train['Response'], train['Previously_Insured'])
cross_tab_previously_insured.columns = ['No Asegurado Prev.', 'Asegurado Prev.']
cross_tab_previously_insured.index = ['No Compraron la Póliza', 'Compraron la Póliza']
print(cross_tab_previously_insured)


sns.countplot(x='Vehicle_Damage', hue='Response', data=train)
plt.xlabel('Daño del Vehículo')
plt.ylabel('Cantidad de Clientes')
plt.title('Distribución de Daño del Vehículo por Respuesta')
plt.legend(title='Respuesta', labels=['No Compraron la Póliza', 'Compraron la Póliza'])
plt.xticks(ticks=[0, 1], labels=['No', 'Sí'])
plt.show()

cross_tab_vehicle_damage = pd.crosstab(train['Response'], train['Vehicle_Damage'])
cross_tab_vehicle_damage.columns = ['Sin Daño', 'Con Daño']
cross_tab_vehicle_damage.index = ['No Compraron la Póliza', 'Compraron la Póliza']
print(cross_tab_vehicle_damage)



sns.countplot(x='Vehicle_Age', hue='Response', data=train)
plt.xlabel('Edad del Vehículo')
plt.ylabel('Cantidad de Clientes')
plt.title('Distribución de la Edad del Vehículo por Respuesta')
plt.legend(title='Respuesta', labels=['No Compraron la Póliza', 'Compraron la Póliza'])
plt.show()


cross_tab_vehicle_age = pd.crosstab(train['Response'], train['Vehicle_Age'])
cross_tab_vehicle_age.index = ['No Compraron la Póliza', 'Compraron la Póliza']
print(cross_tab_vehicle_age)


Variables_categoricas=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

train.head()


plt.figure(figsize=(12, 10)) # Puedes ajustar estos números según tus necesidades
correlation_matrix = train.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


nor = ['Age', 'Vintage']
standard_scaler = StandardScaler().fit(train[nor])
train[nor] = standard_scaler.transform(train[nor])


mm = MinMaxScaler().fit(train[['Annual_Premium']])
train[['Annual_Premium']] = mm.transform(train[['Annual_Premium']])


# Mapeo de género para el conjunto de prueba
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
# Obtener las variables dummies
test = pd.get_dummies(test, drop_first=True)
# Renombrar columnas
test = test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
# Cambio de tipos de datos
test['Vehicle_Age_lt_1_Year'] = test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years'] = test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes'] = test['Vehicle_Damage_Yes'].astype('int')
# Aplicar los mismos escaladores
test[nor] = standard_scaler.transform(test[nor])
test[['Annual_Premium']] = mm.transform(test[['Annual_Premium']])


# Separar las etiquetas
train_target = train['Response']
train = train.drop(['Response'], axis=1)


train.head()


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


#Definir el objeto RandomizedSearchCV con el espacio de búsqueda y el clasificador
random_search = RandomizedSearchCV(tree_clf, param_distributions=param_dist, n_iter=10, cv=4, verbose=0, random_state=42, n_jobs=-1)


# Ajusta el modelo
random_search.fit(train, train_target)


# Evaluar el Modelo
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






# Obtener las probabilidades de la clase positiva
y_probs = best_model.predict_proba(x_test)[:, 1]

# Calcular los valores de la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, label='Curva ROC (área = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--') # Linea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



# Parámetros para la búsqueda
param_dist_rf = {
    'n_estimators': [50, 100, 150],  # Puedes cambiar esto según tus necesidades
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 10],
    'min_samples_leaf': [4, 6, 8],
    'min_samples_split': [5, 7, 10]
}


# Crea un clasificador de Random Forest
rf_clf = RandomForestClassifier()

# Configura la búsqueda aleatoria
random_search_rf = RandomizedSearchCV(rf_clf, param_distributions=param_dist_rf, n_iter=10, cv=4, verbose=1, random_state=42, n_jobs=-1)

# Ajusta el modelo
random_search_rf.fit(x_train, y_train)


# Evaluar el Modelo
print("Mejores parámetros: ", random_search_rf.best_params_)
best_model_rf = random_search_rf.best_estimator_
test_accuracy_rf = best_model_rf.score(x_test, y_test)
print("Precisión en el conjunto de prueba: ", test_accuracy_rf)


# Obtener las probabilidades de la clase positiva
y_probs_rf = random_search_rf.predict_proba(x_test)[:, 1]

# Calcular el AUC-ROC
auc_roc_rf = roc_auc_score(y_test, y_probs_rf)
print("AUC-ROC para Random Forest:", auc_roc_rf)

# Calcular los valores de la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_probs_rf)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr_rf, tpr_rf, label='Curva ROC Random Forest (área = %0.2f)' % auc_roc_rf)
plt.plot([0, 1], [0, 1], 'k--') # Linea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC Random Forest')
plt.legend(loc="lower right")
plt.show()


# Crear un clasificador Gaussian Naive Bayes
gnb = GaussianNB()

# Entrenar el modelo
gnb.fit(x_train, y_train)

# Predicción
y_pred = gnb.predict(x_test)



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


# Obtener las probabilidades de la clase positiva
y_probs = gnb.predict_proba(x_test)[:, 1]

# Calcular el AUC-ROC
auc_roc = roc_auc_score(y_test, y_probs)
print("AUC-ROC:", auc_roc)

# Calcular los valores de la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, label='Curva ROC (área = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--') # Linea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()