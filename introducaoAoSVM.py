from sklearn import datasets
import pandas as pd
import seaborn as sns

iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

iris_df.head()

# Gerar gráfico
sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)',
 'petal length (cm)', 'petal width (cm)', 'species']], hue='species')

from sklearn import svm
from sklearn.model_selection import train_test_split

# Obrigatório fazer o iris = load_iris()
X = iris.data
y = iris.target

X_train, X_teste, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

clf = svm.SVC(C=1.0)

clf.fit(X_train, y_train)

clf.predict(X_teste)

y_pred = clf.predict(X_teste)

clf.score(X_teste, y_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=iris.target_names))

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=13)

n_neighbors = 10

clf = KNeighborsClassifier(n_neighbors=n_neighbors)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

iris = load_iris()
print(classification_report(y_test, y_pred, target_names=iris.target_names))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
