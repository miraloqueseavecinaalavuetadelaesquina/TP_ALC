#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 22:15:02 2025

@author: an2
"""

# =============================================================================
# LIBRERÍAS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx

import os
import sys
path_t40= '/home/an2/git/tp_alc/TP_ALC'
#path_X270 = '/home/kanxo/git/tp_ALC/Anto/'
sys.path.append(path_t40)




# =============================================================================
# FUNCIONES PRELIMINARES
# =============================================================================

# Función para permutar filas (para descompPLU)
def permutacion(A, vector_P, index):
    n = A.shape[0]
    max_index = index + np.argmax(np.abs(A[index:, index]))
    #swap
    if max_index != index:
        A[[index, max_index]] = A[[max_index, index]]
        vector_P[[index, max_index]] = vector_P[[max_index, index]]


# Descomposición PLU con pivoteo
def calculaPLU(m, verbose=False):
    mc = m.copy().astype(np.float64)
    n = m.shape[0]
    P = np.eye(n)
    for i in range(n - 1):
        max_row = i + np.argmax(np.abs(mc[i:, i]))
        if max_row != i:
            mc[[i, max_row]] = mc[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
        a_ii = mc[i, i]
        if a_ii == 0:
            raise ValueError("Matriz singular (no invertible)")
        L_i = mc[i+1:, i] / a_ii
        mc[i+1:, i] = L_i
        mc[i+1:, i+1:] -= np.outer(L_i, mc[i, i+1:])
    
    L = np.tril(mc, -1) + np.eye(n)
    U = np.triu(mc)
    if verbose:
        print("P:\n", P)
        print("L:\n", L)
        print("U:\n", U)
    return P, L, U


def calculaLU(matriz, verbose=False):
    mc = matriz.copy().astype(np.float64)
    n = matriz.shape[0]
    for i in range(n - 1):
        a_ii = mc[i, i]
        if a_ii == 0:
            raise ValueError("Cero en la diagonal durante LU (se requiere pivoteo)")
        L_i = mc[i+1:, i] / a_ii
        mc[i+1:, i] = L_i
        mc[i+1:, i+1:] -= np.outer(L_i, mc[i, i+1:])
    
    L = np.tril(mc, -1) + np.eye(n)
    U = np.triu(mc)
    if verbose:
        print("L:\n", L)
        print("U:\n", U)
    return L, U

# Función para calcular la inversa corregida
def inversa(m):
    n = m.shape[0]
    try:
        L, U = calculaLU(m)
        P = np.eye(n)  # Matriz de permutación identidad si no hay pivoteo
    except (ValueError, LinAlgError):
        P, L, U = calculaPLU(m)
    
    m_inv = np.zeros((n, n))
    for i in range(n):
        e_i = P.T @ np.eye(n)[:, i]  # Aplica la permutación P al vector canónico
        y = solve_triangular(L, e_i, lower=True)
        x = solve_triangular(U, y, lower=False)
        m_inv[:, i] = x
    return m_inv



def calcula_K (A):
    n = A.shape[0]
    k = np.zeros((n,n),dtype=A.dtype)
    for i in range(n):
        k[i][i] = A[i].sum()
    
    return k

def norma2(v):
    n = 0
    for k in v:
        n+=k*k
    return np.sqrt(n)

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)


# =============================================================================
# FUNCIONES TP
# =============================================================================


# L = K -A
def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    L = calcula_K(A) - A
    return L

# P_ij número esperado de conexiones entre i y j
# P_ij = kikj/2E  => P = k . k^t/2E
# R = A - P
def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    k =  np.diag(calcula_K(A)).reshape(-1, 1)
    R = A-  1/np.sum(A) * k @ k.T
    return R

# lambda =(s^t . L . s)/4
def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    # definimos el vector s de signos asociado a v
    s = np.array([-1 if x < 0 else (1 if x > 0 else 0) for x in v])
    lambdon = 1/4 * s.T @ L @ s
    return lambdon

# 
def calcula_Q_(A,R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.array([-1 if x < 0 else (1 if x > 0 else 0) for x in v])
    Q = 1/(2*np.sum(A)) * s.T @ R @ s
    return Q


def potencia(matriz, v0=[], epsilon=1e-8, verbose=True):
    if matriz.shape[0] != len(v0):
        print("dim(v0) != dim(A)")
        return 0    
    if len(v0) == 0: v0 = 2 * np.random.rand(matriz.shape[0],1) - 1    
    r_k =v0.T@matriz@v0/(v0.T@v0) # cociente de raleigh    
    v_k = matriz@v0/np.linalg.norm(matriz@v0,2)
    delta = np.abs(v_k.T@matriz@v_k/(v_k.T@v_k) - r_k)
    
    while delta >= epsilon:
        v_k = matriz@v_k
        v_k/=np.linalg.norm(v_k,2)
        r_temp = v_k.T@matriz@v_k/(v_k.T@v_k)
        delta = np.abs(r_temp-r_k) # otra forma|| v_(k+1) - v_k ||_2
        r_k = r_temp
    r_k = r_k[0][0]
    #v_k = v_k.T[0]
    if verbose :
        print("autovalor : ", r_k)
        print("autovector : \n", v_k)
    return r_k, v_k


def metpot1(A,tol=1e-8,maxrep=np.Inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = 2 * np.random.rand(A.shape[0],1) - 1 # Generamos un vector de partida aleatorio, entre -1 y 1
   v /= np.linalg.norm(v,2) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 /= np.linalg.norm(v1,2) # normalizamos
   l = v.T @ A @ v / (v.T @ v) # autovector estimado A@v = l v1 <=> v*A@v = l v*v <=> l = v*A@v / v*v
   l1 = v1.T @ A @ v1 / (v1.T @ v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v # Calculo nuevo v1
      v1 /= np.linalg.norm(v1,2) # Normalizo
      l1 = v1.T @ A @ v1 / (v1.T @ v1) # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l =  l1[0][0] # Calculamos el autovalor
   return v1,l,nrep<maxrep



# PRE: in A.dtype==float64
def calcularCPA(X, v0=[], cant_componentes=0 , precision=1e-6, ejes=1, verbose=True):
    n = X.shape[ejes]
    cov = np.dot(X.T,X) / n
    if cant_componentes > n or cant_componentes == 0: cant_componentes = n
    if len(v0) == 0: v0 = 2 * np.random.rand(n,1) - 1 
    d_aval = []
    v_avec = []
    n=cant_componentes+1
    while cant_componentes > 0:
        a_val, a_vec = potencia(matriz=cov, v0=v0, epsilon=precision, verbose=False)
        if verbose:
            print("autovalor {}: ".format(n-cant_componentes), a_val)
        cov-= a_val*a_vec@a_vec.T
        d_aval.append(a_val)
        v_avec.append(a_vec.T[0])
        cant_componentes-=1
    v_avec = np.column_stack(v_avec)
    
    return d_aval, v_avec


# deflA = A - l v*v^t/(v^tv)
def deflaciona(A,tol=1e-8,maxrep=np.Inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1/(v1.T @ v1) * np.outer(v1,v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

# 
def metpot2(A,v1,l1,tol=1e-8,maxrep=np.Inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = deflaciona(A)
   return metpot1(deflA,tol,maxrep)

# mu>0, (L+mu*I)^{-1} 
def metpotI(A,mu,tol=1e-8,maxrep=np.Inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    M = inversa(A + mu * np.identity(A.shape[0]))
    return metpot1(M,tol=tol,maxrep=maxrep)


def metpotI2(A,mu,tol=1e-8,maxrep=np.Inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu * np.identity(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = inversa(X) # La invertimos
   v1,l1,_ = metpot1(iX)
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ = metpot1(defliX)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

def particion(A,v):
    # función que me retorna la matriz recortada en dos partes 
    I = np.identity(A.shape[0])
    sp = I[v>0,:] # Particiòn de I para valores positivos de v
    sn = I[v<0,:] # Particiòn de I para valores negativos de v
    Ap = sp @ A @ sp.T
    An = sn @ A @ sn.T

    return Ap, An
    

def laplaciano_iterativo(A,niveles=2,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        mu = 1 # aca deberia ir una funcion para calclar un mu optimo?
        # buscamos el segundo autovecto y autovalor más chico, por lo que usaremos la potencia inversa, tengamos en cuenta
        v,l,_ = metpotI2(L, mu) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        v = v.flatten()
        Ap = A[v>0,:][:,v>0] # Asociado al signo positivo 
        Am = A[v<0,:][:,v<0] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel, caso base
        return([nombres_s])
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        v = v.flatten()
        # Modularidad Actual:
        #Q = calcula_Q_(A, R, v)
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp  = R[v>0,:][:,v>0] # Rp Parte de R asociada a los valores positivos de v
            Rm =  R[v<0,:][:,v<0] # Rm Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm   
            vp = vp.flatten()
            vm = vm.flatten()
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(R=Rp,
                                          nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                    modularidad_iterativo(R=Rm,
                                          nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                    )

# Recibe La matriz A simetrizada
def crear_datos(D,lm=[3] ,niv=2,nombres_s=None, metodo='corte minimo', visualizacion='mapa'):
    # Creamos diccionario
    dict_datos ={'visualizacion':visualizacion,'titulo':metodo , 'numero de conexiones':lm, 'particiones':{}}
    # calculamos particiones para cada metodo
    if metodo == 'corte minimo':
        for m in lm:
            A = calcular_A_simetrica(D,m)
            dict_datos['particiones'][m] = laplaciano_iterativo(A,niveles=niv) 
    elif metodo == 'modularidad':
        for m in lm:
            A = calcular_A_simetrica(D,m)
            dict_datos['particiones'][m] = modularidad_iterativo(A)
            
    return dict_datos


# =============================================================================
# VISUALIZACIÓN
# =============================================================================


def calcular_A_simetrica(D= None,m=3):
    if D is None: print('Dame una matriz valida')
    A = construye_adyacencia(D, m)
    # Simetrizamos
    A = 1/2 * (A + A.transpose())
    return A

def visualizar_comunidades_ax(museos, particiones, ax, titulo="Comunidades de Museos"):
    """
    Visualiza las comunidades de museos en el mapa
    
    Parámetros:
    museos : GeoDataFrame con los datos de los museos
    particiones : Lista de pd.series con índices o nombres de museos por comunidad 
    """
    # Creamos columna de comunidad en el GeoDataFrame copia
    museos_plot = museos.copy()
    museos_plot['comunidad'] = -1  # Valor por defecto
    
    for i, comunidad in enumerate(particiones):
        museos_plot.loc[comunidad, 'comunidad'] = i
    
    # Dibujar barrios
    barrios.boundary.plot(color='gray', ax=ax)
    
    # Graficar museos coloreados por comunidad
    museos_plot.plot(column='comunidad', categorical=True, 
                    legend=False, ax=ax, markersize=50,
                    cmap='tab20') # legend_kwds={'title': "Comunidad"}
    
    ax.set_title(titulo)


def visualizar_comparacion_comunidades(museos, G, G_layout, particiones_metodo1, particiones_metodo2, 
                                      nombre_metodo1='Método 1', nombre_metodo2='Método 2'):
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # para el método 1 
    visualizar_comunidades_ax(museos, particiones_metodo1, axs[0], 
                                titulo=f"{nombre_metodo1} - Comunidades en Mapa")
    
    # para el método 2
    visualizar_comunidades_ax(museos, particiones_metodo2, axs[1], 
                                titulo=f"{nombre_metodo2} - Comunidades en Mapa")
    
    plt.tight_layout()
    plt.show()
        


def visualizar_red_comunidades_ax(G, G_layout, particiones, ax, titulo=''):
    """
    Visualiza la red de museos coloreada por comunidades
    
    Parámetros:
    G : Grafo de NetworkX
    G_layout : Diccionario de posiciones
    particiones : Lista de series con índices de museos por comunidad
    """
    # Asignamos colores a nodos por comunidad
    node_colors = []
    for node in G.nodes():
        for i, comunidad in enumerate(particiones):
            if node in comunidad:
                node_colors.append(i)
                break
    
    # Visualización
    nx.draw_networkx_nodes(G, G_layout, node_color=node_colors, 
                          cmap=plt.cm.tab20, node_size=100, ax=ax)
    nx.draw_networkx_edges(G, G_layout, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, G_layout, font_size=8, ax=ax)
    
    ax.set_title(titulo)
    
    
def visualizar_comparacion_redes(G, G_layout, particiones_metodo1, particiones_metodo2, 
                                nombre_metodo1='Método 1', nombre_metodo2='Método 2'):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # para el método 1
    visualizar_red_comunidades_ax(G, G_layout, particiones_metodo1, axs[0], 
                                    titulo=f"{nombre_metodo1} - Red de Comunidades")
    
    # para el método 2
    visualizar_red_comunidades_ax(G, G_layout, particiones_metodo2, axs[1], 
                                    titulo=f"{nombre_metodo2} - Red de Comunidades")
    
    plt.tight_layout()
    plt.show()



def visualizar_variacion_m(museos, G, G_layout, datos):

    # Visualiza cómo cambian las particiones para diferentes valores de m
    # Obtener los valores de m (máximo 4)
    valores_m = datos['numero de conexiones'][:4]  # Tomar solo los primeros 4 valores
    particiones_dict = datos['particiones']
    
    # Crear figura con 2 columnas y 2 filas
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(datos['titulo'], fontsize=24, y=0.98)
    
    # Aplanamos los ejes una iteración más sencilla
    axs_flat = axs.flatten()
    
    #  (máximo 4)
    for i, m in enumerate(valores_m):
        ax = axs_flat[i]
        particiones = particiones_dict.get(m, [])  # para evitar KeyError
        
        # invocamos la función de visualización
        if datos['visualizacion'] == 'mapa':
            # Visualización geográfica
            museos_copy = museos.copy()
            museos_copy['comunidad'] = -1
            
            # Asignamos comunidad a cada museo
            for comm_idx, comunidad in enumerate(particiones):
                for museo_idx in comunidad:
                    museos_copy.loc[museo_idx, 'comunidad'] = comm_idx
            
            # Dibujar barrios
            barrios.boundary.plot(color='gray', ax=ax)
            
            # Dibujar museos coloreados por comunidad
            museos_copy.plot(column='comunidad', categorical=True, 
                            legend=(i == 0),  # Mostrar leyenda solo en el primer gráfico
                            ax=ax, markersize=30,
                            cmap='tab20', 
                            legend_kwds={'title': "Comunidad"})
            
        elif datos['visualizacion'] == 'red':
            # Visualización de red
            node_colors = []
            for node in G.nodes():
                for comm_idx, comunidad in enumerate(particiones):
                    if node in comunidad:
                        node_colors.append(comm_idx)
                        break
            
            # Dibujar la red
            nx.draw_networkx_nodes(G, G_layout, node_color=node_colors, 
                                  cmap='tab20', node_size=50, ax=ax)
            nx.draw_networkx_edges(G, G_layout, alpha=0.2, ax=ax)
            
            # Dibujar leyenda solo en el primer gráfico
            if i == 0:
                # Crear un mapeo de colores para la leyenda
                from matplotlib.lines import Line2D
                unique_colors = np.unique(node_colors)
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=plt.cm.tab20(color_idx), 
                          markersize=10, label=f'Comunidad {color_idx}')
                    for color_idx in unique_colors
                ]
                ax.legend(handles=legend_elements, title="Comunidades")
        
        # Configurar título del subgráfico
        ax.set_title(f"m = {m}", fontsize=16)
        ax.axis('off') if datos['metodo'] == 'red' else None
    
    # Ocultar ejes vacíos si hay menos de 4 gráficos
    for j in range(len(valores_m), 4):
        axs_flat[j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  
    plt.show()
    return fig


# Carga de datos
# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')

 # visualizacion
# Armamos el gráfico para visualizar los museos
fig, ax = plt.subplots(figsize=(10, 10))
barrios.boundary.plot(color='gray',ax=ax)
museos.plot(ax=ax)

# Matriz de distancias
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()


m = 3 # Cantidad de links por nodo
A = construye_adyacencia(D,m)

# Construcción de la red en NetworkX (sólo para las visualizaciones)
G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
# Construimos un layout a partir de las coordenadas geográficas
G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}

# Visualizacion
fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
nx.draw_networkx(G,G_layout,ax=ax) # Graficamos los museos



# =============================================================================
# TEST
# =============================================================================

A = calcular_A_simetrica(D,5)
k = laplaciano_iterativo(A, niveles=4) #, nombres_s=museos['name']
r = modularidad_iterativo(A) #, nombres_s=museos['name']

visualizar_comparacion_comunidades(museos, G, G_layout, 
                                   particiones_metodo1=k,
                                   particiones_metodo2=r,
                                   nombre_metodo1="Corte mínimo", 
                                   nombre_metodo2="Modularidad")

visualizar_comparacion_redes(G, G_layout,
                             particiones_metodo1 = k,
                             particiones_metodo2 = r,
                             nombre_metodo1="Corte mínimo",
                             nombre_metodo2="Modularidad"
                             )


l = crear_datos(D,lm=[3,5,10,40])
visualizar_variacion_m(museos, G, G_layout, datos=l)

l = crear_datos(D,lm=[3,5,10,40],metodo='modularidad')
visualizar_variacion_m(museos, G, G_layout, datos=l)


# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]])

A = calcula_R(A_ejemplo)

k = calcularCPA(calcula_R(A_ejemplo))

A = np.array([[0,1,2,3],
              [4,0,3,4],
              [2,2,0,12],
              [4,1,8,0]],
             dtype=float)
i = np.identity(4)

r = np.array([1,-0.5,2,-2])
k = i[r<0,:]#[:]   #np.array([[0,1,0,0],[0,0,0,1]])
# r = #np.array([[0,0], [1,0],[0,0],[0,1]])
k @ A @ k.T

A[r<0,:][:,r<0]



v = np.array([1,2,3])
np.outer(v, v)

np.sum(A_ejemplo)

k = calcula_K(A_ejemplo)

l = calcula_L(A_ejemplo)

r = calcula_R(A_ejemplo)

v, l, _ = metpot1(A_ejemplo)
v = v.flatten()
A_ejemplo[v>0,:][:,v>0]

v = np.array([1,-1,0.3,0.45,-0.4, 6, 0.1, -3]).reshape(-1, 1) 

v @ v.transpose()

k = laplaciano_iterativo(A_ejemplo, 1)

r = modularidad_iterativo(A_ejemplo)
