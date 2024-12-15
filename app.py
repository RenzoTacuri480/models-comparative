import src.services.read as rd
from src.services.models import models_data
from src.services.graphics import heatmap, pie_graph, barras_graph, histogram_graph, box_graph
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import pandas as pd
#--------------------------------------------------------------------------------------------

app = Flask(__name__, template_folder='src/templates')
CORS(app)
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
                file.save(filepath)
                print(f"Documento {filepath} guardado con éxito")
            
            data = rd.process_data(filepath)
            #data = pd.read_csv(filepath)

            table_html = data.to_html(classes='table table-bordered table-hover', index=False)

            column_names = data.columns.to_list()
            #return data.to_html()
            print(column_names)
            return render_template('index.html', table_html=table_html, column_names=column_names)
        else:
            return "Debe subir un archivo CSV válido"

    return render_template('index.html', table_html=None, column_names=[])
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
    if request.method == "POST":
        if 'file' not in request.files:
            return "No se ha seleccionado ningún archivo"
        
        file = request.files['file']

        if file and file.filename.endswith('.csv'):
            #Directorio del dataset
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filepath)

            if not os.path.exists(filepath):
                file.save(filepath)
                print(f"Documento {filepath} guardado con éxito")
            
            predictions = pd.read_csv(filepath)

            if data is not None:
                #Lista de predicciones
                model_prediction = models_data(data)
                model_prediction = model_prediction['model'].predict(predictions).tolist()
                labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

                #Cambio de números por etiquetas
                final_results = {
                    'cantidad': len(model_prediction),
                    'info': [labels[val] for val in model_prediction]
                }

                return final_results
            else:
                return "Faltan datos de entrenamiento"
        else:
            return "Debe subir un archivo CSV válido"

    return render_template('predict.html')
#--------------------------------------------------------------------------------------------

#Ejecución final
if __name__ == '__main__':
#    app.run(debug=True, port=5002)
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5002))