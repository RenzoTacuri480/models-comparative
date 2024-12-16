import src.services.read as rd
from src.services.models import models_data
from src.services.graphics import heatmap, pie_graph, barras_graph, histogram_graph, box_graph
from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn import metrics
#--------------------------------------------------------------------------------------------

app = Flask(__name__, template_folder='src/templates')
#--------------------------------------------------------------------------------------------

#Ruta de archivos subidos
UPLOAD_FOLDER = os.path.join(app.root_path, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Datos para analizar
data = None
#--------------------------------------------------------------------------------------------

@app.route('/', methods=["GET", "POST"])
def upload_file():
    global data

    if request.method == "POST":
        if 'file' not in request.files:
            return "No se ha seleccionado ningún archivo"
        
        file = request.files['file']

        if file and file.filename.endswith('.csv'):
            #Directorio del dataset
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filepath)

            if not os.path.exists(filepath):
                temp_data = pd.read_csv(file)
                temp_data.insert(0, 'ID', pd.RangeIndex(start=1, stop=len(temp_data) + 1, step=1))
                
                #file.save(filepath)
                temp_data.to_csv(filepath, index=False)
                print(f"Documento {filepath} guardado con éxito")
            
            data = rd.process_data(filepath)
            column_names = data.columns.to_list()
            #return data.to_html()
            #return jsonify(data.to_dict(orient='records'))
            return render_template('students.html', data=data.to_dict(orient='records'), column_names=column_names)
        else:
            return "Debe subir un archivo CSV válido"

    return render_template('index.html', data=data)
#--------------------------------------------------------------------------------------------

@app.route('/data')
def get_data():
    data_json = []

    if data is not None:
        heatmap(data)

        for i, col in enumerate(data.columns):
            data_json.append({
                'number': i,
                'column': col,
                'type': str(data[col].dtype),
                'corr_target': float(data.corr()['Target'][i])
            })
        
        data_results = {
            'title': 'Correlativos',
            'info': data_json
        }
        return jsonify(data_results)
    else:
        return "No hay datos para analizar"
#--------------------------------------------------------------------------------------------

#Generación de gráficos
@app.route('/graphs')
def graphs_data():
    if data is not None:
        new_data = rd.copy_data(data)

        pie_graph(new_data)
        barras_graph(new_data)
        histogram_graph(new_data)
        box_graph(new_data)

        return "Gráficos creados con éxito"
    else:
        return "No hay datos para analizar"
#--------------------------------------------------------------------------------------------

@app.route('/models')
def analysis_data():
    if data is not None:
        #Métricas y gráficos de modelos predictivos
        m_results = models_data(data)

        return jsonify(m_results['results'])
    else:
        return "No hay datos para analizar"
#--------------------------------------------------------------------------------------------

@app.route('/predict', methods = ["GET", "POST"])
def predict_data():
    data = pd.read_csv("static/dataset.csv")
    new_data = rd.copy_data(data)

    X = new_data.drop('Target', axis=1)
    y = new_data['Target']

    #Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rfc = RandomForestClassifier(random_state=2)

    #Entrenamiento
    rfc.fit(X_train, y_train)

    models_results = []
    results = {}

    y_predictions = {}

    #Cálculo de métricas por modelo
    models = [
        ("Random Forest", rfc)
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

        #Usado también para elegir mejor modelo
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

    try:
        data_value = request.get_json()
        print(f"ID usado: {data_value['ID']}")

        predict_value = pd.read_csv("static/test.csv")

        row_data = predict_value[predict_value['ID'] == int(data_value['ID'])]
        row_data = row_data.drop(columns=['ID'])
        print(f"row_data: {row_data}")

        #predictions = row_data.values.tolist()
        #print(f"Array predictions: {predictions}")
        #row_data = row_data.fillna(0)
        #row_data = row_data.values.reshape(1, -1)
        row_data.to_csv("static/valor.csv", index=False)
        valor = pd.read_csv("static/valor.csv")

        print("---------")

        #model_prediction = models_data()
        #print(f"Resultados: {model_prediction['results']}")
        #model_prediction = model_prediction['model'].predict(valor).tolist()
        model_prediction = rfc.predict(valor).tolist()
        labels = {0: 'Deserción', 1: 'Inscrito', 2: 'Graduado'}

        #Cambio de números por etiquetas
        final_results = {
            'cantidad': len(model_prediction),
            'info': [labels[val] for val in model_prediction]
        }
        print(final_results)

        #return jsonify(final_results)
        return render_template('predict.html', results=final_results)
    
    except Exception as e:
        return str(e), 400


@app.route('/analisis')
def analisis():
    return render_template('analisis.html') 

@app.route('/models_info')
def models_info():
    return render_template('models.html') 
#--------------------------------------------------------------------------------------------

#Ejecución final
if __name__ == '__main__':
#    app.run(debug=True, port=5002)
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5002))