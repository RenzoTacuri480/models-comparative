from flask import Flask, render_template, jsonify, url_for, request, redirect, flash
#----------------------------------------------------------------------------
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import os
#----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn import metrics
#----------------------------------------------------------------------------
app = Flask(__name__)

#Lectura del Dataset de Dropout
data = pd.read_csv("app/dataset.csv")

#data.head()
data.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)
data.isnull().sum()/len(data)*100
data['Target'] = data['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})
print(data["Target"].unique())
data.corr()['Target']
#----------------------------------------------------------------------------

@app.route('/info')
def get_info():

    data_json = []

    for i, col in enumerate(data.columns):
        #Obtener tipo de dato
        dtype = str(data[col].dtype)

        data_json.append({
            'number': i,
            'column': col,
            'dtype': dtype,
        })
    
    data_info = {
        'title': 'Información del dataset',
        'info': data_json,
        'cantidad': len(data_json)
    }

    #print(data_info)
    return render_template('info.html', data = data_info)

@app.route('/data')
def get_data():
    
    data_json = []

    #Generación de heatmap
    plt.figure(figsize=(30, 30))
    sns.heatmap(data.corr() , annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('static/heatmap.png', bbox_inches='tight')

    for i, col in enumerate(data.columns):

        data_json.append({
            'number': i,
            'column': col,
            'corr_target': float(data.corr()['Target'][i])
        })
    
    data_results = {
        'title': 'Información del dataset',
        'info': data_json,
        'url_image': url_for('static', filename='heatmap.png')
    }

    print(data_results['url_image'])
    return render_template('data.html', data = data_results)

def copy_data():
    new_data = data.copy()
    new_data = new_data.drop(columns=['Nationality',
                                    'Mother\'s qualification',
                                    'Father\'s qualification',
                                    'Educational special needs',
                                    'International',
                                    'Curricular units 1st sem (without evaluations)',
                                    'Unemployment rate',
                                    'Inflation rate'], axis=1)
    #new_data.info()
    new_data['Target'].value_counts()
    return new_data

def pie_graph(new_data):
    #Gráfico de pastel
    x = new_data['Target'].value_counts().index
    y = new_data['Target'].value_counts().values

    df = pd.DataFrame({
        'Target': x,
        'Count_T' : y
    })

    fig = px.pie(df,
                names ='Target',
                values ='Count_T',
                title='¿Cuántos Desertores, Matriculados y Graduados hay en la columna de la Variable Objetivo?')

    fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
    fig.write_image('static/pie.png', width=900, height=600)

def barras_graph():
    #Gráfico de barras
    correlations = data.corr()['Target']
    top_10_features = correlations.abs().nlargest(10).index
    top_10_corr_values = correlations[top_10_features]

    plt.figure(figsize=(10, 11))
    plt.bar(top_10_features, top_10_corr_values)
    plt.xlabel('Columnas')
    plt.ylabel('Correlación con la Variable Objetivo')
    plt.title('Top 10 columnas con mayor correlación con la Variable Objetivo')
    plt.xticks(rotation=45)
    plt.savefig('static/top_barras.png', bbox_inches='tight')

def histograma(new_data):
    #Histograma
    fig_hist = px.histogram(new_data['Age'], x='Age',color_discrete_sequence=['lightblue'])
    fig_hist.write_image('static/histogram.png', width=800, height=600)

def box_graph(new_data):
    #Gráfico de caja
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='Age', data=new_data)
    plt.xlabel('Target')
    plt.ylabel('Age')
    plt.title('Relación entre la edad y la Variable Objetivo')
    plt.savefig('static/box.png', bbox_inches='tight')

@app.route('/images')
def images_data():
    new_data = copy_data()

    pie_graph(new_data)
    barras_graph()
    histograma(new_data)
    box_graph(new_data)

    data_images = {
        'title': 'Imágenes generadas',
        'pie': url_for('static', filename = 'pie.png'),
        'top_barras': url_for('static', filename = 'top_barras.png'),
        'histogram': url_for('static', filename = 'histogram.png'),
        'box': url_for('static', filename = 'box.png')
    }

    print(data_images)
    return render_template('images.html', data = data_images)

@app.route('/models')
def analysis_data():
    new_data = copy_data()

    X = new_data.drop('Target', axis=1)
    y = new_data['Target']

    #Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    dtree = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=2)
    lr = LogisticRegression(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    abc = AdaBoostClassifier(n_estimators=50,learning_rate=1, random_state=0)
    sm = svm.SVC(kernel='linear', probability=True)


    dtree.fit(X_train,y_train)
    rfc.fit(X_train,y_train)
    lr.fit(X_train,y_train)
    knn.fit(X_train,y_train)
    abc.fit(X_train, y_train)
    sm.fit(X_train, y_train)

    models_results = []
    results = {}

    # Diccionario para almacenar las predicciones
    y_predicciones = {}

    # Predicciones y cálculo de métricas para cada modelo
    modelos = [
        ("Decision Tree", dtree),
        ("Random Forest", rfc),
        ("Logistic Regression", lr),
        ("K-Nearest Neighbors", knn),
        ("AdaBoost", abc),
        ("SVM", sm)
    ]

    for nombre, modelo in modelos:

        # Predicción
        y_pred = modelo.predict(X_test)

        # Guardar predicciones para uso futuro
        y_predicciones[nombre] = y_pred

        # Cálculo de métricas
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
        recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
        f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)

        models_results.append({
            'name': nombre,
            'accuracy': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })

        results[nombre] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

    df_resultados = pd.DataFrame(results).T

    # Configuramos el tamaño del gráfico
    plt.figure(figsize=(10, 6))

    # Creamos el gráfico de barras
    df_resultados.plot(kind='bar', figsize=(12, 6), colormap='viridis')

    # Añadimos título y etiquetas
    plt.title('Comparación de Métricas por Modelo', fontsize=16)
    plt.ylabel('Porcentaje (%)', fontsize=12)
    plt.xlabel('Modelos', fontsize=12)

    # Rotamos las etiquetas del eje x
    plt.xticks(rotation=45)

    # Mostramos el gráfico
    plt.tight_layout()
    plt.savefig('static/models_barras.png', bbox_inches='tight')

    #Creación de las matrices de confusión
    confusion_matrices = [
        (metrics.confusion_matrix(y_test, y_predicciones.get("Decision Tree")), "Decision Tree"),
        (metrics.confusion_matrix(y_test, y_predicciones.get("Random Forest")), "Random Forest"),
        (metrics.confusion_matrix(y_test, y_predicciones.get("Logistic Regression")), "Logistic Regression"),
        (metrics.confusion_matrix(y_test, y_predicciones.get("K-Nearest Neighbors")), "K-Nearest Neighbors"),
        (metrics.confusion_matrix(y_test, y_predicciones.get("AdaBoost")), "AdaBoost"),
        (metrics.confusion_matrix(y_test, y_predicciones.get("SVM")), "SVM")
    ]

    class_names = ['Dropout', 'Enrolled', 'Graduate']
    models_images = []

    for cm, model_name in confusion_matrices:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names)

        # Título y etiquetas
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')

        models_images.append({
            'image': plt.savefig(f'static/models_matrix_{model_name.replace(" ", "_")}.png', bbox_inches='tight')
        })

    resultados = {
        'title': 'Comparativa de modelos',
        'info': models_results,
        'barras': url_for('static', filename = 'models_barras.png'),
        'matrix': models_images
    }

    print(resultados)
    return render_template('models.html', data = resultados)

@app.route('/')
def index():
    return render_template('menu.html')

if __name__ == '__main__':
    app.run(debug = True)