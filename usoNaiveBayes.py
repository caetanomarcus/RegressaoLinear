import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('Data/insurance.csv')
base['Accident'] = base['Accident'].fillna('None')

base.drop(['Unnamed: 0'], axis=1, inplace=True)
#print(base.shape)
#print(base.head())

#variavels idependente como y (minusculo) e independentes como X (maiusculo)

y = base.iloc[:, 7].values
X = base.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].values

labelEncoder = LabelEncoder()

for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = labelEncoder.fit_transform(X[:, i])

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)

modelo = GaussianNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)

print(previsoes)

accuracy = accuracy_score(y_teste, previsoes)

precision = precision_score(y_teste, previsoes, average=None)
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')

report = classification_report(y_teste, previsoes)

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
print(report)

confusao = ConfusionMatrix(modelo, classes=['None', 'Severe', 'Mild', 'Moderate'])
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.show()
