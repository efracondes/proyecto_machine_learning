# En este archivo .py se encuentran las funciones mas importantes que se 
# ocupan a lo largo de los notebooks. 

# Llamamos a las siguientes bibliotecas
#from IPython import get_ipython
import pandas as pd
import numpy as np
import math
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm_notebook as tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

############################## EPSILONS #################################
# ESTAS SON LAS FUNCIONES PARA CALCULAR LAS EPSILONS, PARA MAS INFO
# SOBRE EL SIGNIFICADO DE ESTA FUNCION POR FAVOR REFIERASE AL REPORTE
# DEL PROYECTO.

def N_total(X):
    return len(X.index)

def N_C(Y,valor_clase):
    N_o = 0
    for persona in Y.index:
        if Y[persona] == valor_clase:
            N_o = N_o + 1
    return N_o

def N_X(X,variable,valor_variable):
    condicion = X[variable] == valor_variable
    return len(X[condicion])

def N_CX(X,Y,variable,valor_variable,valor_clase):
    count = 0
    for persona in X.index:
        if X.at[persona,variable] == valor_variable:
            if Y[persona] == valor_clase:
                count = count + 1
    
    return count

def epsilon(N,Nc,Nx,Ncx):
    
    Pc = Nc / N
    Pcx = Ncx / Nx
    
    numerador = math.sqrt(Nx) * (Pcx -Pc)
    denominador = math.sqrt( Pc * (1 - Pc) )
    
    epsi = numerador / denominador
    
    return epsi

def epsilon_variable_y_valor(X,Y,variable,valor_variable,valor_clase):
    
    N = N_total(X)
    Nc = N_C(Y,valor_clase)
    Nx = N_X(X,variable,valor_variable)
    Ncx = N_CX(X,Y,variable,valor_variable,valor_clase)
    
    if Nx == 0:
        return 0
    
    else:
        return epsilon(N,Nc,Nx,Ncx)

def valores_de_variable(X,variable):
    valores = [] # aqui guardaremos los valores
    
    for persona in X.index:
        valor = X.at[persona,variable] 
        
        if valor not in valores:
            valores.append(valor)
    
    return valores

def diccionario_valores_de_variables(X):

    valores = {}
    variables = list(X.columns)
    
    for variable in variables:
        valores[variable] = valores_de_variable(X,variable)
        
    return valores

def epsilons_de_datos(X,Y,valor_clase):
    epsilons = {}
    epsilons['variable'] = []
    epsilons['valor'] = []
    epsilons['Nx'] = []
    epsilons['Nxc'] = []
    epsilons['N'] = []
    epsilons['Nc'] = []
    epsilons['Pc'] = []
    epsilons['Pcx'] = []
    epsilons['epsilon'] = []
    
 
    valores = diccionario_valores_de_variables(X)
    variables = list(valores.keys())

    for variable in tqdm(variables):
        for valor in valores[variable]:
        
            epsilons['valor'].append(valor)
            epsilons['variable'].append(variable)
            
            N = N_total(X)
            Nc = N_C(Y,valor_clase)
            Nx = N_X(X,variable,valor)
            Nxc = N_CX(X,Y,variable,valor,valor_clase)
            Pc = Nc / N
            Pcx = Nxc / Nx
            
            epsilons['Nx'].append(Nx)
            epsilons['Nxc'].append(Nxc)
            epsilons['N'].append(N)
            epsilons['Nc'].append(Nc)
            epsilons['Pc'].append(Pc)
            epsilons['Pcx'].append(Pcx)
        
            epsi = epsilon_variable_y_valor(X,Y,variable,valor,valor_clase)
            epsilons['epsilon'].append(epsi)

    return pd.DataFrame(epsilons)

def variables_importantes(epsilons,cuantos):
    
    variables = []
    epsilons_importantes = epsilons.sort_values( by = 'epsilon', ascending = False )
    count = 0
    
    while len(variables) < cuantos:
        variable = epsilons_importantes.iloc[count]['variable']
        
        if variable not in variables:
            variables.append(variable)
            
        count = count + 1
    
    return variables

def categorias(X,variable):
    categorias_lista = []
    
    for i in X.index:
        categoria = X.at[i,variable]
        if categoria not in categorias_lista:
            categorias_lista.append(categoria)
    
    return categorias_lista 


###### FUNCIONES PARA EL ANALISIS DE ERRORES Y DESEMPENO #####

def porcentaje_de_aciertos(f,y):
    f = list(f)
    y = list(y)
    aciertos = 0
    n = len(y)
    for i in range(n):
        if f[i] == y[i]:
            aciertos = aciertos + 1
    
    return aciertos * 100 / n   

def error_cuadratico(f,y):
    E = ((f - y)**2)
    E = np.array(E)
    return E.mean()


# %%
def cuenta_repeticiones(lista,elemento):
    count = 0
    for li in lista:
        if li == elemento:
            count = count + 1
    return count

def verdaderos_positivos(f,y):
    verdaderos_positivos_count = 0
    for i in range(len(f)):
        if f[i] == 1 and f[i] == y[i]:
            verdaderos_positivos_count = verdaderos_positivos_count + 1
    return verdaderos_positivos_count
    
def falsos_positivos(f,y):
    falsos_positivos_count = 0
    for i in range(len(f)):
        if f[i] == 1 and f[i] != y[i]:
            falsos_positivos_count = falsos_positivos_count + 1
    return falsos_positivos_count

def tasa_verdaderos_positivos(f,y):
    elementos_relevantes = cuenta_repeticiones(y,1)
    return verdaderos_positivos(f,y) / elementos_relevantes

def tasa_falsos_positivos(f,y):    
    elementos_irrelevantes = cuenta_repeticiones(y,0)
    return falsos_positivos(f,y) / elementos_irrelevantes

def exhaustividad(f,y):
    return tasa_verdaderos_positivos(f,y)

def precision(f,y):
    elementos_seleccionados = cuenta_repeticiones(f,1)
    return verdaderos_positivos(f,y) / elementos_seleccionados

def area_bajo_la_curva(X,Y):
    return metrics.auc(X,Y)

def graficar_ROC(probabilidades,y):
    probabilidades = list(probabilidades)
    y = list(y)
    s_min = min(probabilidades)
    s_max = max(probabilidades)
    tasas_de_verdaderos_positivos = []
    tasas_de_falsos_positivos = []
    umbrales = np.linspace(s_min,s_max,10000)
    
    for umbral in umbrales:
        f = prediccion_NB(probabilidades,umbral)
        tasas_de_verdaderos_positivos.append(tasa_verdaderos_positivos(f,y))
        tasas_de_falsos_positivos.append(tasa_falsos_positivos(f,y))
        
    plt.plot(tasas_de_falsos_positivos,tasas_de_verdaderos_positivos)
    plt.grid(True)
    plt.title('Curva ROC')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    
    area = area_bajo_la_curva(tasas_de_falsos_positivos,tasas_de_verdaderos_positivos)
    print('El area bajo la curva ROC es = '+str(area))

def graficar_ROC_scores(probabilidades,y):
    probabilidades = list(probabilidades)
    y = list(y)
    s_min = min(probabilidades)
    s_max = max(probabilidades)
    tasas_de_verdaderos_positivos = []
    tasas_de_falsos_positivos = []
    umbrales = np.linspace(s_min,s_max,5000)
    
    for umbral in umbrales:
        f = predicciones_scores(probabilidades,umbral)
        tasas_de_verdaderos_positivos.append(tasa_verdaderos_positivos(f,y))
        tasas_de_falsos_positivos.append(tasa_falsos_positivos(f,y))
        
    plt.plot(tasas_de_falsos_positivos,tasas_de_verdaderos_positivos)
    plt.grid(True)
    plt.title('Curva ROC')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    
    area = area_bajo_la_curva(tasas_de_falsos_positivos,tasas_de_verdaderos_positivos)
    print('El area bajo la curva ROC es = '+str(area))
    
def promedio_error_y_aciertos_modelo(datos,columna_de_la_clase,iteraciones, modelo):

    error_modelo = 0
    porcentaje_aciertos_modelo = 0

    for i in tqdm(range(iteraciones)):
        
        datos_train , datos_test = dividir_datos(datos,0.7)
    
        X_train = datos_train.drop(columns = [columna_de_la_clase])
        Y_train = datos_train[columna_de_la_clase]
        X_test = datos_test.drop(columns = [columna_de_la_clase])
        Y_test = datos_test[columna_de_la_clase]
        
        if modelo == 'logistic_regression':
            clasificador = LogisticRegression(solver='newton-cg',penalty='l2')
        elif modelo == 'random_forest':
            # Obtenido en el notebook de random forest
            N_arboles = 8000
            clasificador = RandomForestClassifier(n_estimators=N_arboles, random_state=0)
        
        clasificador.fit(X_train,Y_train)
    
        f_test = clasificador.predict(X_test)
        f_train = clasificador.predict(X_train)
        
        
        error_iteracion = error_cuadratico(f_test,Y_test) # calculamos el error cuadratico
        porcentaje_aciertos_iteracion = proyecto_ml.porcentaje_de_aciertos(f_test,Y_test) 
    
        error_modelo = error_modelo + error_iteracion
        porcentaje_aciertos_modelo = porcentaje_aciertos_modelo + porcentaje_aciertos_iteracion

    print('El modelo predijo en promedio correctamente el ' + str(porcentaje_aciertos_modelo/iteraciones) + '% de los datos de test.')
    print('El modelo tuvo en promedio un error cuadratico medio del ' + str(error_modelo/iteraciones) + '% en los datos de test.')   

############## FUNCIONES PARA REALIZAR VALIDACION CRUZADA ##############

def dividir_datos(datos,porcentaje):
    datos_train, datos_test = train_test_split(datos, train_size = 0.7)
    return datos_train, datos_test
    
# Función para calcular el error cuadratico medio de una sola iteración

def cross_validation_one_iteration_logistic_regression(datos_tot, columna_de_la_clase):
    
    datos_train , datos_test = dividir_datos(datos_tot,0.7)
    
    X_train = datos_train.drop(columns = [columna_de_la_clase])
    Y_train = datos_train[columna_de_la_clase]
    X_test = datos_test.drop(columns = [columna_de_la_clase])
    Y_test = datos_test[columna_de_la_clase]
    
    regresion = LogisticRegression(solver='newton-cg',penalty='l2',C=0.005)
    regresion.fit(X_train,Y_train)
    
    f_test = regresion.predict(X_test)
    f_train = regresion.predict(X_train)
    
    error_test = error_cuadratico(f_test,Y_test)
    error_train = error_cuadratico(f_train,Y_train)
    
    return error_test , error_train

N_arboles = 2000

def cross_validation_one_iteration_random_forest(datos_tot, columna_de_la_clase):
    
    datos_train , datos_test = dividir_datos(datos_tot,0.7)

    X_train = datos_train.drop(columns = [columna_de_la_clase])
    Y_train = datos_train[columna_de_la_clase]
    X_test = datos_test.drop(columns = [columna_de_la_clase])
    Y_test = datos_test[columna_de_la_clase]
    # Obtenido en el notebook de random forest

    bosque = RandomForestClassifier(n_estimators=N_arboles, random_state=0)
    bosque.fit(X_train,Y_train)
    
    f_train = bosque.predict(X_train)
    f_test = bosque.predict(X_test)
    
    error_test = error_cuadratico(f_test,Y_test)
    error_train = error_cuadratico(f_train,Y_train)
    
    return error_test , error_train

def cross_validation(datos_tot, columna_de_la_clase, iteraciones, modelo):

    error_cuadratico_test = []
    error_cuadratico_train = []
    
    if modelo == 'logistic_regression':
        cross_validation_one_iteration = cross_validation_one_iteration_logistic_regression
    elif modelo == 'random_forest':
        cross_validation_one_iteration = cross_validation_one_iteration_random_forest
    
    for i in range(iteraciones):
        error_test , error_train = cross_validation_one_iteration(datos_tot, columna_de_la_clase)
        
        error_cuadratico_test.append(error_test)
        error_cuadratico_train.append(error_train)
        
    error_cuadratico_test = np.array(error_cuadratico_test)
    error_cuadratico_train = np.array(error_cuadratico_train)
    
    
    return error_cuadratico_test.mean() , error_cuadratico_train.mean()

def seleccion_de_variables(datos, seleccion_de_variables, columna_de_la_clase, modelo):
    error_validacion = np.inf
    variables = list(datos.columns)
    variables.remove(columna_de_la_clase)
    
    for variable in variables:
        if variable not in seleccion_de_variables: # no tomamos en cuenta variables que ya esten en el arreglo
            
            datos_seleccionados = datos[ seleccion_de_variables + [variable,columna_de_la_clase] ]
            e_test , e_train = cross_validation(datos_seleccionados, columna_de_la_clase, 20, modelo) # numero de particiones
            #print(variable)
            
            if e_test < error_validacion:
                error_validacion = e_test
                error_entrenamiento = e_train
                variable_mas_importante = variable
                
    nueva_seleccion_de_variables = seleccion_de_variables + [variable_mas_importante]
    
    # devuelve una lista con las variables y el error producido de las predicciones con esas variables
    return nueva_seleccion_de_variables , error_validacion , error_entrenamiento 