
train = pd.read_csv('train.csv')


print("Datos de entrenamiento:", train.shape) 

train.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)
