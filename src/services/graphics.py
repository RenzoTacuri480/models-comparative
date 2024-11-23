from src.services.models import models, df_results, y_test, y_predictions

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import metrics

#Generación de Heatmap
def heatmap(data):
    plt.figure(figsize=(30, 30))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('src/images/heatmap.png', bbox_inches='tight')
#--------------------------------------------------------------------------------------------

#Gráfico de pastel
def pie_graph(data):
    x = data['Target'].value_counts().index
    y = data['Target'].value_counts().values

    df = pd.DataFrame({
        'Target': x,
        'Count_T': y
    })

    fig = px.pie(df, names='Target', values='Count_T',
                 title='¿Cuántos Desertores, Matriculados y Graduados hay en la Variable Objetivo?')
    
    fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4, textinfo='value+label', pull=[0, 0.2, 0.1])
    fig.show()
    fig.write_image('src/images/pie.png', width=900, height=600)
#--------------------------------------------------------------------------------------------

#Gráfico de barras
def barras_graph(data):
    correlations = data.corr()['Target']
    top_10_features = correlations.abs().nlargest(10).index
    top_10_corr_values = correlations[top_10_features]

    plt.figure(figsize=(10, 11))
    plt.bar(top_10_features, top_10_corr_values)
    plt.xlabel('Columnas')
    plt.ylabel('Correlación con la Variable Objetivo')
    plt.title('Top 10 columnas con mayor correlación con la Variable Objetivo')
    plt.xticks(rotation=45)
    plt.savefig('src/images/top_barras.png', bbox_inches='tight')
#--------------------------------------------------------------------------------------------

def histogram_graph(data):
    fig_hist = px.histogram(data['Age'], x='Age', color_discrete_sequence=['lightblue'])
    fig_hist.show()
    fig_hist.write_image('src/images/histogram.png', width=800, height=600)
#--------------------------------------------------------------------------------------------

#Gráfico de caja
def box_graph(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='Age', data=data)
    plt.xlabel('Target')
    plt.ylabel('Age')
    plt.title('Relación entre la edad y Variable Objetivo')
    plt.savefig('src/images/box.png', bbox_inches='tight')
#--------------------------------------------------------------------------------------------

#Métricas por modelo
def models_metrics():
    plt.figure(figsize=(10, 6))

    df_results.plot(kind='bar', figsize=(12, 6), colormap='viridis')

    #Descripción gráfica
    plt.title('Comparación de métricas por modelo', fontsize=16)
    plt.ylabel('Porcentaje (%)', fontsize=12)
    plt.xlabel('Modelos', fontsize=12)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('src/images/models_barras.png', bbox_inches='tight')
#--------------------------------------------------------------------------------------------

#Matrices de confusión
def models_matrix():
    matrices = []

    for name, model in models:
        matrices.append((metrics.confusion_matrix(y_test, y_predictions.get(name)), name))

    class_names = ['Dropout', 'Enrolled', 'Graduate']

    for cm, model_name in matrices:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
    
        #Descripción gráfica
        plt.title(f"Matriz de confusión - {model_name}")
        plt.ylabel('Valor real')
        plt.xlabel('Valor predecido')

        plt.savefig(f'src/images/model_matrix_{model_name.replace(" ", "_")}.png', bbox_inches='tight')