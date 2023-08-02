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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



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


#RANDOM FOREST
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

#Naive bayes
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

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=0)
mm = MinMaxScaler()
x_train[['Vintage']] = mm.fit_transform(x_train[['Vintage']])
x_test[['Vintage']] = mm.transform(x_test[['Vintage']])


# Create PyTorch tensors for train and test data
x_train_tensor = torch.tensor(x_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
x_test_tensor = torch.tensor(x_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))

# Create DataLoader for batching
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Define neural network model
class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_shape, 24)
        self.fc2 = nn.Linear(24, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    

    # Instantiate the model
model = Model(x_train.shape[1])

# Define loss and optimizer
ratio = y_train.value_counts()[1] / len(y_train)
class_weights = torch.tensor([1 - ratio, ratio - 0.1])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters())


# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

# Pasada hacia adelante en los datos de prueba
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    predicted_labels = torch.argmax(test_outputs, dim=1)

# Calcular la precisión
precisión = (predicted_labels == y_test_tensor).float().mean()
print(f'Precisión: {precisión.item()}')

# Calcular la pérdida usando la función CrossEntropyLoss
pérdida = criterion(test_outputs, y_test_tensor)
print(f'Pérdida: {pérdida.item()}')
