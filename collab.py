
train = pd.read_csv('C:/Users/monte/Desktop/tesis/train.csv')
test = pd.read_csv('C:/Users/monte/Desktop/tesis/test.csv')

train.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)
test.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)

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

# Create PyTorch tensors for train and test data
x_train_tensor = torch.tensor(x_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
x_test_tensor = torch.tensor(x_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))

# Create DataLoader for batching
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_shape, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x


# Instantiate the model
model = Model(x_train.shape[1])

# Define the class weights and convert to Float
class_0_weight = 1 / y_train.value_counts()[0]
class_1_weight = 1 / y_train.value_counts()[1]
class_weights = torch.tensor([class_0_weight, class_1_weight], dtype=torch.float)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Estableciendo tasa de aprendizaje a 0.01
# Training loop
EPOCHS = 4

for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

# Forward pass on test data
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    predicted_labels = torch.argmax(test_outputs, dim=1)

# Convertir los datos de prueba en un tensor
x_test_tensor = torch.tensor(test.values.astype(np.float32))

# Pasar los datos de prueba a través del modelo
with torch.no_grad():
    test_outputs = model(x_test_tensor)

# Obtener las etiquetas predichas
predicted_labels = torch.argmax(test_outputs, dim=1)

import torch.nn.functional as F

# Asegúrate de poner tu modelo en modo de evaluación
model.eval()

# Suponiendo que x_test_tensor contiene tus datos de prueba
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    
    # Aplicar la función Softmax para convertir las salidas en probabilidades
    probabilities = F.softmax(test_outputs, dim=1)
    
    # Obtener las etiquetas predichas (índice de la probabilidad máxima)
    predicted_labels = torch.argmax(probabilities, dim=1)


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
