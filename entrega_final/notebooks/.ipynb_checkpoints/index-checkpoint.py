import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Dask dependencias
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import dask.bag as db
import time
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
import numpy as np
import json
import geopandas as gpd
from fuzzywuzzy import fuzz
import re
from wordcloud import WordCloud
from prefect import flow, task

@task(name = "iniciar cliente dask")
def iniciar_cliente_dask():
    cliente = Client("tcp://scheduler:8786")
    cliente 

@task(name = "leer data de csv")
def leer_data():
    print("----> leyendo datos...")
    remote_path = './data.csv'
    # Lee el archivo CSV utilizando Dask y el cliente conectado
    df = dd.read_csv(remote_path, blocksize="20MB", encoding='latin-1', low_memory=False, dtype={'Fecha_muerte': 'object', 'Pais_viajo_1_nom': 'object', 'Pais_viajo_1_cod': 'object', 'per_etn_': 'float64'})
    print("----> datos leídos...")
    return df
    
@task(name = "computar data")
def computed_data(df):
    print("----> computando datos...")
    full_data_computed = df.compute()
    print(full_data_computed)
    return full_data_computed

@task(name = "contar na")
def contar_na(full_data):
    print(full_data.isna().sum())

@task(name = "eliminar na")
def eliminar_na(full_data_computed):
    full_data_computed.dropna()
    full_data_computed = full_data_computed.drop(columns=['Caso', 'Fecha Not', 'Departamento', 'Ciudad_municipio', 'Fecha_inicio_sintomas', 'Fecha_diagnostico', 'Fecha_muerte', 'Pais_viajo_1_cod', 'Pais_viajo_1_nom', 'Fecha_recuperado', 'nom_grupo_', 'fecha_hoy_casos'])
    print(full_data_computed)

@task(name = "obtener departamentos")
def obtener_departamentos(full_data_computed):
    ruta_archivo = './colombia.geo.json'
    # Lee el archivo GeoJSON utilizando geopandas
    departamentos = gpd.read_file(ruta_archivo)
    return departamentos

@task(name = "limpiando data")
def limpiando_data(full_data, full_data_computed):
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('BOGOTA', 'SANTAFE DE BOGOTA D.C')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('BARRANQUILLA', 'ATLANTICO')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('VALLE', 'VALLE DEL CAUCA')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('STA MARTA D.E.', 'MAGDALENA')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('CARTAGENA', 'BOLIVAR')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('NORTE SANTANDER', 'NORTE DE SANTANDER')
    full_data_computed['Departamento_nom'] = full_data_computed['Departamento_nom'].replace('SAN ANDRES', 'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA')
    
    # Función para encontrar la mejor coincidencia aproximada
    def encontrar_coincidencia(valor, opciones):
        mejor_puntuacion = -1
        mejor_opcion = None

        # Verificar si el valor es "en estudio" sin eliminar espacios en blanco
        if valor.lower() == 'en estudio':
            return 'en estudio'

        valor = re.sub(' +', ' ', valor.strip())  # Eliminar espacios en blanco adicionales

        for opcion in opciones:
            puntuacion = fuzz.ratio(valor, opcion)
            if puntuacion > mejor_puntuacion:
                mejor_puntuacion = puntuacion
                mejor_opcion = opcion

        return mejor_opcion

    
    def aplicar_transformacion(columna, opciones):
        columna = columna.astype(str)

        # Convertir a minúsculas
        columna = columna.str.lower()

        # Eliminar caracteres especiales y puntuación
        columna = columna.str.replace('[^\w\s]', '', regex=True)

        # Reemplazar "N/A" con "Desconocido"
        columna = columna.replace('n/a', 'desconocido')

        # Reemplazar "" con "Desconocido"
        columna = columna.replace('', 'desconocido')

        # Reemplazar valores con la mejor coincidencia aproximada
        columna = columna.apply(lambda x: encontrar_coincidencia(x, opciones))
        return columna

    def apply_transformations(df):
        opciones_por_columna = {
            'Sexo': ['f', 'm'],
            'Fuente_tipo_contagio': ['comunitaria', 'relacionado', 'en estudio', 'importado'],
            'Ubicacion': ['casa', 'fallecido', 'hospital'], 
            'Estado': ['leve', 'moderado', 'grave', 'fallecido'],
            'Recuperado': ['recuperado', 'fallecido', 'activo'],
            'Tipo_recuperacion': ['tiempo', 'pcr']
        }

        for columna, opciones in opciones_por_columna.items():
            df[columna] = aplicar_transformacion(df[columna], opciones)

        return df

    # Aplicar las transformaciones utilizando map_partitions
    full_data = full_data.map_partitions(apply_transformations)

    full_data_computed = full_data.compute()
    # Imprimir el resultado
    print(full_data_computed)

@task(name = "visualizar data mapas")
def visualizar_data_mapas(full_data_computed, departamentos):
    # Obtener el recuento de casos por departamento
    recuento_departamentos = full_data_computed['Departamento_nom'].value_counts().reset_index()
    recuento_departamentos.columns = ['Departamento_nom', 'Recuento_departamento']

    # Combinar los datos geoespaciales de los departamentos con el recuento de casos
    mapa_departamentos = departamentos.merge(recuento_departamentos, left_on='NOMBRE_DPT', right_on='Departamento_nom')

    # Generar la visualización del mapa de casos por departamento
    fig, ax = plt.subplots(figsize=(12, 8))
    mapa_departamentos.plot(ax=ax, column='Recuento_departamento', cmap='Reds', linewidth=0.8, edgecolor='0.8', legend=True)
    ax.set_title('Recuento de casos por departamento en Colombia')
    plt.show()

@task(name = "ver palabras departamentos")
def ver_palabras_departamentos(full_data_computed):
    # Especificar el nombre de la columna que contiene las palabras
    columna_palabras = 'Ciudad_municipio_nom'

    # Obtener el recuento de palabras y convertirlo a un diccionario de frecuencias
    recuento_palabras = full_data_computed[columna_palabras].value_counts().to_dict()

    # Crear el objeto WordCloud con las palabras y sus frecuencias
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(recuento_palabras)

    # Configurar el tamaño de las palabras en la nube de palabras
    wordcloud.recolor(random_state=42)

    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Principales ciudades por número de afectados')
    plt.show()

@task(name = "dibujar kmeans")
def dibujar_kmeans(full_data_computed):
    sample_for_kmeans = full_data_computed.sample(frac=0.0001)
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
    categoricos_data = encoder.fit_transform(sample_for_kmeans)
    
    # Combinación de atributos categóricos y numéricos
    combined_features = np.concatenate((sample_for_kmeans, categoricos_data), axis=1)
    
    # Calcular la inercia para diferentes valores de k en K-means,utilizo datos categoricos, datos numericos genera error.
    inertias = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(categoricos_data)
        inertias.append(kmeans.inertia_)
    
    # Visualizar la curva de la inercia
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo (Elbow Method)')
    plt.show()

@flow(name = "flujo_principal")
def main():
    print("--------------------> LECTURA DE BASE DE DATOS")
    print("----> iniciando cliente de dask")
    ## iniciar_cliente_dask()
    print("----> leer_data")
    full_data = leer_data()
    full_data_computed = computed_data(full_data)
    print("----> contar nulos o na")
    #contar_na(full_data_computed)
    print("----> eliminar na y columnas inutiles para el análisis")
    eliminar_na(full_data_computed)
    print("--------------------> PRE-PROCESAMIENTO DE LOS DATOS")
    print("----> limpiando data (puede tardar unos minutos)")
    #limpiando_data(full_data, full_data_computed)
    print("----> obtener departamentos")
    departamentos = obtener_departamentos(full_data_computed)
    print("--------------------> VISUALIZAR DATOS")
    print("----> imprimiendo mapas")
    visualizar_data_mapas(full_data_computed, departamentos)
    print("----> ver palabras departamentos")
    ver_palabras_departamentos(full_data_computed)
    print("--------------------> MACHINE LEARNING")
    print("----> dibujar kmeans")
    dibujar_kmeans(full_data_computed)

    
    # print("----> obtener_categoricos")
    # categoricos_data = obtener_categoricos(full_data)
    # print("----> visualizar_curva_inercias")
    # visualizar_curva_inercias(full_data.computed, categoricos_data)
    # print("----> imprimir_clusters")
    # imprimir_clusters(categoricos_data)


if __name__ == '__main__':
    start_time = time.time()
    print("----> Starting process ...")
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("----> Ending process")
    