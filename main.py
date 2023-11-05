import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Clonando o repositório com o dataset

! git clone https://github.com/renadeveloper/course_betech_modelos_produtivos_I/

# Carregando o dataset

shoppers_intention = pd.read_csv('./course_betech_modelos_produtivos_I/online_shoppers_intention.csv', sep=';',decimal=',')

# Removendo dados duplicados

shoppers_intention.drop_duplicates(inplace=True)

# Encodando as variáveis objeto

label_encoder = preprocessing.LabelEncoder()
shoppers_intention['Revenue']= label_encoder.fit_transform(shoppers_intention['Revenue'])

shoppers_intention['Weekend']= label_encoder.fit_transform(shoppers_intention['Weekend'])

shoppers_intention= pd.get_dummies(shoppers_intention, columns = ['Month','VisitorType'])

 Agora iremos transformar as variáveis que deveriam ser numérica com astype

shoppers_intention['Administrative_Duration'] = shoppers_intention['Administrative_Duration'].str.replace('.', '').astype(float)

shoppers_intention['BounceRates'] = shoppers_intention['BounceRates'].astype(float)

shoppers_intention['ExitRates'] = shoppers_intention['ExitRates'].astype(float)

shoppers_intention['SpecialDay'] = shoppers_intention['SpecialDay'].astype(float)

shoppers_intention['PageValues'] = shoppers_intention['PageValues'].str.replace('.', '').astype(float)

shoppers_intention['ProductRelated_Duration'] = shoppers_intention['ProductRelated_Duration'].str.replace('.', '').astype(float)

shoppers_intention['Informational_Duration'] = shoppers_intention['Informational_Duration'].str.replace('.', '').astype(float)

# Separando os dados de treino e teste

X = shoppers_intention.drop(columns=['Revenue'], axis=0)
y = shoppers_intention['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Use class_weight='balanced' para balancear as classes

clf_random = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_random = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1,  random_state=42)

# Treinando o modelo

clf_random.fit(X_train, y_train)

# Obtendo as previsões

y_pred = clf_random.predict(X_test)
