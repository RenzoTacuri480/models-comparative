from src.services.read import data, copy_data
from src.services.models import models_results
from src.services.graphics import heatmap, pie_graph, barras_graph, histogram_graph, box_graph, models_metrics, models_matrix
from flask import Flask, render_template, jsonify, url_for, request, redirect, flash
#--------------------------------------------------------------------------------------------

app = Flask(__name__)
#--------------------------------------------------------------------------------------------

@app.route('/info')
def get_info():
    data_json = []

    for i, col in enumerate(data.columns):
        #Obtención de tipos de dato
        dtype = str(data[col].dtype)

        data_json.append({
            'number': i,
            'column': col,
            'type': dtype
        })

    data_info = {
        'title': 'Columnas del dataset',
        'info': data_json,
        'cantidad': len(data_json)
    }

    return jsonify(data_info)
#--------------------------------------------------------------------------------------------

@app.route('/data')
def get_data():
    data_json = []

    heatmap(data)

    for i, col in enumerate(data.columns):
        data_json.append({
            'number': i,
            'column': col,
            'corr_target': float(data.corr()['Target'][i])
        })
    
    data_results = {
        'title': 'Correlativos',
        'info': data_json
    }
    
    return jsonify(data_results)
#--------------------------------------------------------------------------------------------

#Generación de gráficos
@app.route('/graphs')
def graphs_data():
    new_data = copy_data(data)

    pie_graph(new_data)
    barras_graph(new_data)
    histogram_graph(new_data)
    box_graph(new_data)

    return "Gráficos creados con éxito"
#--------------------------------------------------------------------------------------------

@app.route('/models')
def analysis_data():
    #Uso de los modelos de predicción (métricas)
    models_metrics()

    #Matrices de confusión
    models_matrix()

    m_results = models_results

    return jsonify(m_results)
#--------------------------------------------------------------------------------------------

#Ejecución final
if __name__ == '__main__':
    app.run(debug=True, port=5002)