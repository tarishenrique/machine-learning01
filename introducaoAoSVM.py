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

clf.score(X_teste, y_test)