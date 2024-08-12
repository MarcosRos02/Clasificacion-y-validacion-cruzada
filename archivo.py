# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Librerias usadas
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Directorio
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
data = pd.read_csv('C:/Users/Marcos/Desktop/TP2 LDD/emnist_letters_tp.csv', header = None)
data = pd.read_csv("~/Documentos/Labo/Cursadas/Labo de Datos/tp2/emnist_letters_tp.csv", header= None)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#       FUNCIONES ÚTILES
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Función para subsetear un data frame a columnas que tengan valores distintos de 0
def filtrar_nulos(datos):
    columnas_a_eliminar = []
    for i in datos.columns:
        if (datos[i] == 0).all():
            columnas_a_eliminar.append(i)
    datos = datos.drop(columnas_a_eliminar, axis = 1)
    return datos


# Funcion  para rotar letras para visualizar
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Limpieza de datos
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data = data.rename(columns={data.columns[0]: 'Letras'}) # Renombrar la primera columna como 'Letras'
nombre_pixeles = {data.columns[i]: f'pixel_{i}' for i in range(1,785)} # Crear diccionario con los nombres de los píxeles
data = data.rename(columns = nombre_pixeles) # Renombrar la columnas de los píxeles con sus nombres

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Análisis exploratorio
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Cantidad de datos
cant_de_datos = data.shape[0] # Shape devuelve en la primera posición [0] la cantidad de filas y en la segunda posición [1] la cantidad de columnas

# Cantidad y tipos de atributos
# Tipo de atributos: Númericos
cantidad_columnas = data.shape[1] 
cant_de_atributos = cantidad_columnas - 1 # Los atributos son todas las columnas menos la variable de interés

# Cantidad de clases de la variable de interés (letras)
letras_sin_repetidos = sql^"""
                            SELECT DISTINCT Letras
                            FROM data
                            ORDER BY Letras ASC
                           """
cant_de_clases = letras_sin_repetidos.shape[0] # Cantidad de filas

# Distribución de muestras de cada letra
distribution = data['Letras'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
distribution.plot(kind='bar')
plt.title('Distribución de las letras')
plt.xlabel('Letras')
plt.ylabel('Frecuencia')
plt.show()

# Promedio global de todas las letras
promedio_global = data.mean()

# Seleccionamos la letra "C" para observar la consistencia intraletra
data_letra_c = sql^"""
                   SELECT *
                   FROM data 
                   WHERE Letras = 'C'"""

# Vemos la variabilidad con el desvío estandar:

desvios_c = data_letra_c.std(axis=0).to_frame().T
desvios_globales = data.std(axis=0).to_frame().T

image_array = np.array(desvios_c).astype(np.float32)

plt.imshow(flip_rotate(image_array))
plt.title("Desvios estandar por pixel - Letra C")
plt.axis('off')  
plt.show()

image_array = np.array(desvios_globales).astype(np.float32)

plt.imshow(flip_rotate(image_array))
plt.title("Desvios estandar por pixel - gobales")
plt.axis('off')  
plt.show()

# Seleccion de letras E, M y L para explorar similitud/diferencia
# interletra, tomando la letra promedio, de los pixeles que no son 
# universalmente 0 en todas las filas. 

data_sin_ceros_absolutos = filtrar_nulos(data)
data_letra_E = sql^"""
                            SELECT *
                            FROM data_sin_ceros_absolutos
                            WHERE Letras = 'E'
                       """
data_letra_E = data_letra_E.mean()
data_letra_E.name = "Letra E"

data_letra_M = sql^"""
                            SELECT *
                            FROM data_sin_ceros_absolutos
                            WHERE Letras = 'M'
                       """

data_letra_M = data_letra_M.mean()

data_letra_M.name = "Letra M"

data_letra_L = sql^"""
                            SELECT *
                            FROM data_sin_ceros_absolutos
                            WHERE Letras = 'L'
                       """

data_letra_L = data_letra_L.mean()

data_letra_L.name = "Letra L"

data_letras_a_compararar = pd.DataFrame([data_letra_E,  data_letra_L, data_letra_M]).T

# Grafico scatter de los pixeles de una letra vs la otra para 
# observar similitud

# Letras E y M

fig, ax = plt.subplots()


ax.scatter(x= "Letra E",
           y= "Letra M",
           data = data_letras_a_compararar)
# Seteo limites para que la  diagonaol sea mas claraen un grafico 
# "cuadrado"
ax.set_xlim(0,220)
ax.set_ylim(0,220)
ax.set_title("Relacion entre pixeles")
ax.set_xlabel("Letra E")
ax.set_ylabel("Letra M")
#%%
# Letras E y L

fig, ax = plt.subplots()


ax.scatter(x= "Letra E",
           y= "Letra L",
           data = data_letras_a_compararar)
# Seteo limites para que la  diagonaol sea mas claraen un grafico 
# "cuadrado"
ax.set_xlim(0,220)
ax.set_ylim(0,220)
ax.set_title("Relacion entre pixeles")
ax.set_xlabel("Letra E")
ax.set_ylabel("Letra L")


# Letra promedio de cada letra
promedio_de_cada_letra = data.groupby('Letras').mean().reset_index() # Realiza un groupby para agrupar cada letra y hace un promedio de cada pixel (dejaba la columna 'Letras' como index, por eso reseteamos el index)

# Calcular la moda del promedio de cada letra (No se realizó gráfico ya que era igual al de moda de cada una de las letras)
moda_promedio = promedio_de_cada_letra.drop('Letras', axis = 1).mode(axis=1) # Descarta la columna 'Letras' y calcula la moda por cada letra
moda_promedio = pd.concat([promedio_de_cada_letra['Letras'], moda_promedio], axis = 1) # Agrega la columna 'Letras'
moda_promedio.columns = ['Letras', 'moda']

letras = moda_promedio['Letras'].tolist()
moda = moda_promedio['moda'].tolist()

# Graficar la moda por letra promedio
plt.bar(letras, moda, color ='skyblue', edgecolor = 'black')

plt.xlabel('Letras')
plt.ylabel('Moda')
plt.title('Moda por letra')

plt.xticks(rotation= 45)

plt.show()

# Calcular la mediana del promedio de cada letra
mediana_promedio = promedio_de_cada_letra.drop('Letras', axis = 1).median(axis = 1) # Descarta la columna 'Letras' y calcula la mediana de cada letra
mediana_promedio = pd.concat([promedio_de_cada_letra['Letras'], mediana_promedio], axis = 1) # Agrega la columna 'Letras'
mediana_promedio.columns = ['Letras', 'mediana']

letras = mediana_promedio['Letras']
mediana_intensidad_pixel = mediana_promedio['mediana']

# Graficar la mediana por letra promedio
plt.bar(letras, mediana_intensidad_pixel, color='skyblue', edgecolor='black')

plt.xlabel('Letras')
plt.ylabel('Mediana ')
plt.title(' Mediana por letra')

plt.xticks(rotation=45)

plt.show()

# Calcular la moda de cada una de las muestras de letras (No se realizó gráfico ya que era igual al calculado anteriormente)
moda_muestras = data.drop('Letras', axis = 1).mode(axis = 1)
moda_muestras = pd.concat([data['Letras'], moda_muestras], axis = 1)
moda_muestras.columns = ['Letras', 'moda']

letras = moda_muestras['Letras']
moda = moda_muestras['moda']

# Calcular la mediana de cada una de las muestras de letras
mediana_muestras = data.drop('Letras', axis = 1).median(axis = 1)
mediana_muestras = pd.concat([data['Letras'], mediana_muestras], axis = 1)
mediana_muestras.columns = ['Letras', 'mediana']

letras = mediana_muestras['Letras']
mediana = mediana_muestras['mediana']

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Clasificacion Binaria
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# i) A partir del dataframe original, construir un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a las letras L o A.

# Subconjunto de imágenes correspondientes a las letras A y L
subconjunto_data = sql^"""
                            SELECT *
                            FROM data
                            WHERE Letras = 'A' OR Letras = 'L'
                       """
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ii) Sobre este subconjunto de datos, analizar cuántas muestras se tienen y determinar si está balanceado con respecto a las dos clases a predecir (la imagen es de la letra L o de la letra A). 
# Cantidad de muestras
cant_muestras = subconjunto_data.shape[0]

# Cantidad de muestras A
subconjunto_de_A = sql^"""
                            SELECT *
                            FROM subconjunto_data
                            Where Letras = 'A'
                        """
cant_muestras_A = subconjunto_de_A.shape[0]

# Cantidad de muestras B
subconjunto_de_L = sql^"""
                            SELECT *
                            FROM subconjunto_data
                            Where Letras = 'L'
                        """    
cant_muestras_L = subconjunto_de_L.shape[0]

# Gráfico de muestras de A y L

plt.bar(['A','L'], [cant_muestras_A, cant_muestras_L])

plt.xlabel('Letra')
plt.ylabel('Cantidad de muestras')
plt.title('Balance de muestras')

plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# iii) Separar los datos en conjuntos de train y test.

#       SEPARACIÓN DE DATOS EN TRAIN Y TEST

# Atributos
X = subconjunto_data.drop('Letras', axis = 1)
# Variable de interés
y = subconjunto_data[['Letras']]
# Separar el 80% para train y el 20% para test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# iv) Ajustar un modelo de KNN en los datos de train, considerando pocos atributos, por ejemplo 3. Probar con distintos conjuntos de 3 atributos y comparar resultados. 
# Analizar utilizando otras cantidades de atributos. Para comparar los resultados de cada modelo usar el conjunto de test generado en el punto anterior.

#       FUNCIONES ÚTILES

# Función para eliminar los elementos repetidos de una lista
def eliminar_repetidos(lista):
    res = []
    for elemento in lista:
        if elemento not in res:
            res.append(elemento)
    return res

# Función para eliminar los pixeles que a nuestro criterio no son claves ( intensidad del pixel < 1)
def eliminar_pixeles(df):
    res = []
    for columna in df:
        if df[columna].values >= 1:
            res.append(columna)
    return res

#       PREPARACIÓN DE LOS DATOS

# Obtener los pixeles "claves" de la letra A 
letra_promedio_A = sql^"""
                         SELECT *
                         FROM promedio_de_cada_letra
                         WHERE Letras = 'A'
                       """

letra_promedio_A = letra_promedio_A.drop('Letras', axis = 1)

# Lista de pixeles claves de A
pixeles_clave_A = eliminar_pixeles(letra_promedio_A) 


# Obtener los pixeles "claves" de la letra L
letra_promedio_L = sql^"""
                         SELECT *
                         FROM promedio_de_cada_letra
                         WHERE Letras = 'L'
                       """
                       
letra_promedio_L = letra_promedio_L.drop('Letras', axis = 1)

# Lista de pixeles claves de L
pixeles_clave_L = eliminar_pixeles(letra_promedio_L)

# Conjunto de pixeles claves de las letras A y L
pixeles_clave = eliminar_repetidos(pixeles_clave_L + pixeles_clave_A)


#        MODELO KNN (3 atributos)

# Generar 5 conjuntos de 3 atributos clave seleccionando aleatoriamente
conjuntos_de_3_atributos = [np.random.choice(pixeles_clave, size = 3, replace = False).tolist() for _ in range(0,5)] 

# Lista para almacenar los resultados de los conjuntos de 3 atributos
resultados_3_atributos = []

# modelo KNN para diferentes conjuntos de 3 atributos
for atributo in conjuntos_de_3_atributos:
    # Utilizar conjunto de 3 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3) # modelo en abstracto con 3 vecinos
    model.fit(X_train_subset, y_train) # entreno mi modelo con los datos de train
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar la exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_3_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })


#        MODELO KNN (10 atributos)

# Generar 5 conjuntos de 10 atributos clave seleccionando aleatoriamente
conjuntos_de_10_atributos = [np.random.choice(pixeles_clave, size=10, replace=False ).tolist() for _ in range(0,5)]

# Lista para almacenar los resultados de los conjuntos de 10 atributos
resultados_10_atributos = []

# modelo KNN para diferentes conjuntos de 10 atributos
for atributo in conjuntos_de_10_atributos:
    # Utilizar conjunto de 10 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subset, y_train)
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar su exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_10_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })


#        MODELO KNN (50 atributos)

# Generar 5 conjuntos de 50 atributos clave seleccionando aleatoriamente
conjuntos_de_50_atributos = [np.random.choice(pixeles_clave, size=50, replace=False ).tolist() for _ in range(0,5)]

# Lista para almacenar los resultados de los conjuntos de 50 atributos
resultados_50_atributos = []

# modelo KNN para diferentes conjuntos de 50 atributos
for atributo in conjuntos_de_50_atributos:
    # Utilizar conjunto de 50 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subset, y_train)
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar su exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_50_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })


#        MODELO KNN (100 atributos)

# Generar 5 conjuntos de 100 atributos clave seleccionando aleatoriamente
conjuntos_de_100_atributos = [np.random.choice(pixeles_clave, size=100, replace=False ).tolist() for _ in range(0,5)]

# Lista para almacenar los resultados de los conjuntos de 100 atributos
resultados_100_atributos = []

# modelo KNN para diferentes conjuntos de 100 atributos
for atributo in conjuntos_de_100_atributos:
    # Utilizar conjunto de 100 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subset, y_train)
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar su exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_100_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })


#        MODELO KNN (300 atributos)

# Generar 5 conjuntos de 300 atributos clave seleccionando aleatoriamente
conjuntos_de_300_atributos = [np.random.choice(pixeles_clave, size=300, replace=False ).tolist() for _ in range(0,5)]

# Lista para almacenar los resultados de los conjuntos de 300 atributos
resultados_300_atributos = []

# modelo KNN para diferentes conjuntos de 300 atributos
for atributo in conjuntos_de_300_atributos:
    # Utilizar conjunto de 300 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subset, y_train)
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar su exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_300_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })
    

#        MODELO KNN (500 atributos)

# Generar 5 conjuntos de 500 atributos clave seleccionando aleatoriamente
conjuntos_de_500_atributos = [np.random.choice(pixeles_clave, size=500, replace=False ).tolist() for _ in range(0,5)]

# Lista para almacenar los resultados de los conjuntos de 500 atributos
resultados_500_atributos = []

# modelo KNN para diferentes conjuntos de 500 atributos
for atributo in conjuntos_de_500_atributos:
    # Utilizar conjunto de 500 atributos
    X_train_subset = X_train[atributo]
    X_test_subset = X_test[atributo]
    
    # Entrenar el modelo de KNN con 3 vecinos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subset, y_train)
    
    # Predecir en el conjunto de test
    y_pred = model.predict(X_test_subset)
    
    # Evaluar su exactitud
    exactitud = accuracy_score(y_test, y_pred)
    
    # Guardar las exactitudes por conjunto de atributos
    resultados_500_atributos.append({
        'atributos': atributo,
        'exactitud': exactitud
        })

    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Visualizacion de exactitudes por cantidad de atributos
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# cantidad de atributos (eje x)
cant_atributos = [3, 10, 50, 100, 300, 500]

# exactitudes (eje y)

# promedio de exactitud de cada conjunto de diferentes cantidad de atributos
prom_exact_3_atributos = np.mean([resultado['exactitud'] for resultado in resultados_3_atributos])
prom_exact_10_atributos = np.mean([resultado['exactitud'] for resultado in resultados_10_atributos])
prom_exact_50_atributos = np.mean([resultado['exactitud'] for resultado in resultados_50_atributos])
prom_exact_100_atributos = np.mean([resultado['exactitud'] for resultado in resultados_100_atributos])
prom_exact_300_atributos = np.mean([resultado['exactitud'] for resultado in resultados_300_atributos])
prom_exact_500_atributos = np.mean([resultado['exactitud'] for resultado in resultados_500_atributos])

# Lista que contiene el promedio de exactitud por conjunto de cantidad de atributo
exactitudes = [prom_exact_3_atributos, prom_exact_10_atributos, prom_exact_50_atributos, prom_exact_100_atributos, 
               prom_exact_300_atributos, prom_exact_500_atributos]

# Crear gráfico
plt.plot(cant_atributos, exactitudes, marker = "o")
plt.title('Exactitud por cantidad de atributos')
plt.xlabel('Cantidad atributos')
plt.ylabel('Exactitud (accuracy)')
plt.grid(True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# v) Comparar modelos de KNN utilizando distintos atributos y distintos valores de k (vecinos). Para el análisis de los resultados,
# tener en cuenta las medidas de evaluación (por ejemplo, la exactitud) y la cantidad de atributos.

# cantidad de vecinos a probar (del 1 al 50)
cant_de_vecinos = range(1,51)

#        MODELO KNN (3 atributos) para diferentes k

# Lista para almacenar los atributos y sus exactitudes con sus k
res_3_atributos = []

# modelo KNN para diferentes 3 atributos y diferentes k (de 1 a 50)
for atributos in conjuntos_de_3_atributos:
    # Utilizar los conjuntos de 3 atributos
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    # Iterar los k-vecinos
    for k in cant_de_vecinos:
        
        # Entrenar el modelo KNN para diferentes k (1;50)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_subset, y_train)
        
        # Predecir las clases para el conjunto de test
        y_pred = model.predict(X_test_subset)
        
        # Evaluar su exactitud
        exactitud = accuracy_score(y_test, y_pred)
        
        # Guardar los resultados
        res_3_atributos.append({
            'k': k,
            'exactitud': exactitud
            })

# Convertir a dataframe y luego obtener una lista con las exactitudes promedio por k
res_3_atributos = pd.DataFrame(res_3_atributos)
exact_3_atributos = res_3_atributos.groupby('k').mean('exactitud').reset_index()['exactitud'].tolist()


#        MODELO KNN (50 atributos) para diferentes k

# Lista para almacenar los atributos y sus exactitudes con sus k
res_50_atributos = []

# modelo KNN para diferentes 50 atributos y diferentes k (de 1 a 50)
for atributos in conjuntos_de_50_atributos:
    # Utilizar los conjuntos de 50 atributos
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    # Iterar los k-vecinos
    for k in cant_de_vecinos:
        
        # Entrenar el modelo KNN para diferentes k (1;50)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_subset, y_train)
        
        # Predecir las clases para el conjunto de test
        y_pred = model.predict(X_test_subset)
        
        # Evaluar su exactitud
        exactitud = accuracy_score(y_test, y_pred)
        
        # Guardar los resultados
        res_50_atributos.append({
            'k': k,
            'exactitud': exactitud
            })  

# Convertir a dataframe y luego obtener una lista con las exactitudes promedio por k
res_50_atributos = pd.DataFrame(res_50_atributos)
exact_50_atributos = res_50_atributos.groupby('k').mean('exactitud').reset_index()['exactitud'].tolist()


#        MODELO KNN (100 atributos) para diferentes k
        
# Lista para almacenar los atributos y sus exactitudes con sus k
res_100_atributos = []

# modelo KNN para diferentes 100 atributos y diferentes k (de 1 a 50)
for atributos in conjuntos_de_100_atributos:
    # Utilizar los conjuntos de 100 atributos
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    # Iterar los k-vecinos
    for k in cant_de_vecinos:
        
        # Entrenar el modelo KNN para diferentes k (1;50)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_subset, y_train)
        
        # Predecir las clases para el conjunto de test
        y_pred = model.predict(X_test_subset)
        
        # Evaluar su exactitud
        exactitud = accuracy_score(y_test, y_pred)
        
        # Guardar los resultados
        res_100_atributos.append({
            'k': k,
            'exactitud': exactitud
            })  

# Convertir a dataframe y luego obtener una lista con las exactitudes promedio por k
res_100_atributos = pd.DataFrame(res_100_atributos)
exact_100_atributos = res_100_atributos.groupby('k').mean('exactitud').reset_index()['exactitud'].tolist()


#        MODELO KNN (300 atributos) para diferentes k

# Lista para almacenar los atributos y sus exactitudes con sus k
res_300_atributos = []

# modelo KNN para diferentes 300 atributos y diferentes k (de 1 a 50)
for atributos in conjuntos_de_300_atributos:
    # Utilizar los conjuntos de 300 atributos
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    # Iterar los k-vecinos
    for k in cant_de_vecinos:
        
        # Entrenar el modelo KNN para diferentes k (1;50)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_subset, y_train)
        
        # Predecir las clases para el conjunto de test
        y_pred = model.predict(X_test_subset)
        
        # Evaluar su exactitud
        exactitud = accuracy_score(y_test, y_pred)
        
        # Guardar los resultados
        res_300_atributos.append({
            'k': k,
            'exactitud': exactitud
            })  

# Convertir a dataframe y luego obtener una lista con las exactitudes promedio por k
res_300_atributos = pd.DataFrame(res_300_atributos)
exact_300_atributos = res_300_atributos.groupby('k').mean('exactitud').reset_index()['exactitud'].tolist()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#       Visualización de exactitud por cantidad de vecinos para diferentes cantidades de atributos
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Crear gráfico
plt.plot(cant_de_vecinos,exact_50_atributos,marker="*")
plt.plot(cant_de_vecinos,exact_100_atributos,marker="o")
plt.plot(cant_de_vecinos,exact_300_atributos,marker="v")
plt.title('Exactitud por cantidad de vecinos (k)')
plt.xlabel('k')
plt.ylabel('Exactitud (accuracy)')
plt.xticks(ticks=cant_de_vecinos[1::2])
plt.legend(['50 atrib.', '100 atrib.', '300 atrib.'])
plt.grid(True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Clasificacion Multiclase
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# i) Vamos a trabajar con los datos correspondientes a las 5 vocales. Primero filtrar solo los datos correspondientes a esas letras. Luego, separar el conjunto de datos en desarrollo (dev) y validación (held-out).
# Para los incisos b y c, utilizar el conjunto de datos de desarrollo. Dejar apartado el conjunto held-out en estos incisos.

# DataFrame solo con vocales
data_con_vocales = sql^"""
                        SELECT *
                        FROM data
                        WHERE Letras = 'A' OR Letras = 'E' OR Letras = 'I' OR Letras = 'O' OR Letras = 'U'
                       """

# atributos
x = data_con_vocales.drop(['Letras'], axis = 1)
# variable de interés
y = data_con_vocales['Letras']     
# Separar el subconjunto de datos en desarrollo y validación (80% dev y 20% eval)
x_dev, x_eval, y_dev, y_eval = train_test_split(x,y, random_state = 42,test_size=0.2)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ii) Ajustar un modelo de árbol de decisión. Probar con distintas profundidades.
 
# Diferentes profundidades del arbol
profundidades = [1,2,3,5,10,13,15, 20, 25, 45]
# Separar el desarrollo en conjuntos de test y train
x_dev_train, x_dev_test, y_dev_train, y_dev_test = train_test_split(x_dev, y_dev, random_state=42, test_size=0.2)
#%%

# Lista para almacenar los resultados
result = []

for profundidad in profundidades:
    model = DecisionTreeClassifier(max_depth= profundidad)
    model.fit(x_dev_train, y_dev_train)
    
    y_dev_pred = model.predict(x_dev_test)
    
    exactitud = accuracy_score(y_dev_test, y_dev_pred)
    
    result.append({
        'profundidad': profundidad,
        'exactitud': exactitud
        })

result = pd.DataFrame(result)
profundidad = result['profundidad']
exactitud = result['exactitud']
#%%
# Graficar las exactitudes en función de la profundidad del árbol de decisión
plt.plot(profundidad, exactitud, marker='D')

plt.title('Exactitud por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Exactitud (accuracy)')

plt.xticks(profundidad, fontsize= 8)

plt.grid(True)

plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# iii) Realizar un experimento para comparar y seleccionar distintos árboles de decisión, con distintos hiperparámetos. Para esto, utilizar validación cruzada con k-folding.
# ¿Cuál fue el mejor modelo? Documentar cuál configuración de hiperparámetros es la mejor, y qué performance tiene.

# Lista de hiperparametros para luego combinar
hiperparametros = {
    'profundidad': [1,3,5,10,13,15],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto',0.5,0.7,None ] # cantidad de características que evalua
    }

# configurar el kfold para la validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state = 42)

# Guardar la mejor combinación de hiperparametros
mejor_modelo = None

# Guardar la mejor exactitud
mejor_exactitud = 0

# Guardar la exactitud actual
actual_exactitud = 0

# Combinar los hiperparametros
for profundidad in hiperparametros['profundidad']:
    for max_feature in hiperparametros['max_features']:
        for criterio in hiperparametros['criterion']:
            # Modelo actual
            model = DecisionTreeClassifier(max_depth = profundidad, max_features= max_feature, criterion = criterio)
            
            # Performance actual del modelo
            actual_exactitud = cross_val_score(model, x_dev, y_dev, cv=kf, scoring='accuracy').mean() # se encarga de ajustar el modelo y evaluar su rendimiento en cada fold
            
            if actual_exactitud > mejor_exactitud:
                mejor_exactitud = actual_exactitud
                mejor_modelo = model

print("\n")
print("Mejor modelo:", mejor_modelo)
print("Y su exactitud:", mejor_exactitud)            
print("\n")           

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# iv) Entrenar el modelo elegido a partir del inciso previo, ahora en todo el conjunto de desarrollo. Utilizarlo para predecir las clases en el conjunto held-out y reportar la performance.

# Entrenar el modelo elegido en todo el conjunto de desarrollo
mejor_modelo.fit(x_dev, y_dev)

# Predecir las clases del conjunto held out
y_pred_held_out = mejor_modelo.predict(x_eval)

# Reportar la performance
exactitud = accuracy_score(y_eval, y_pred_held_out)
#%%
# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_eval, y_pred_held_out)
#%%
# Obtener las clases de mi modelo
clases_unicas = y_eval.unique()

# Visualizarlas
print("Clases únicas:")
for clase in clases_unicas:
    print(clase)
#%%
# Calcular los tipos de errores para las clases
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)

# Imprimir los resultados
for i in range(len(TP)):
    print(f"Clase {i}:")
    print(f"Falsos positivos (FP): {FP[i]}")
    print(f"Falsos negativos (FN): {FN[i]}")
    print(f"Verdaderos positivos (TP): {TP[i]}")
    print(f"Verdaderos negativos (TN): {TN[i]}")

print("\n")  
print("clase 0: U, clase 1: I, clase 2: A, clase 3: E, clase 4: O")
