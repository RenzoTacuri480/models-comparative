from src.services.read import data, copy_data
import pandas as pd

#Modelos de predicción
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

new_data = copy_data(data)

X = new_data.drop('Target', axis=1)
y = new_data['Target']

#Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dtree = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=2)
lr = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
sm = svm.SVC(kernel='linear', probability=True)

#Entrenamiento
dtree.fit(X_train, y_train)
rfc.fit(X_train, y_train)
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
abc.fit(X_train, y_train)
sm.fit(X_train, y_train)

models_results = []
results = {}

y_predictions = {}

#Cálculo de métricas por modelo
models = [
    ("Decision Tree", dtree),
    ("Random Forest", rfc),
    ("Logistic Regression", lr),
    ("K-Nearest Neighbors", knn),
    ("AdaBoost", abc),
    ("SVM", sm)
]

for name, model in models:
    #Predicción
    y_pred = model.predict(X_test)

    y_predictions[name] = y_pred

    #Méticas de importancia
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)

    models_results.append({
        'name': name,
        'accuracy': accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    })

    results[name] = {
        "Accuracy": accuracy,
        "Presicion": precision,
        "Recall": recall,
        "F1-score": f1
    }

df_results = pd.DataFrame(results).T