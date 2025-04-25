#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:54:45 2025

@author: kanxo
"""

# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

import os
import sys
__file__= '/home/kanxo/git/tp_ALC/Anto/funciones.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import funciones as f
import func as ff
import fun as fff

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
# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()

# Matriz de adbyacencia
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

#--------------------------------TP-----------------------------------------------

"""
Usando la factorización LU implementada, encuentre el vector p = M^{−1} b en los si-
guientes casos:
    
a. Construyendo la red conectando a cada museo con sus m = 3 vecinos más cercanos,
calculen el Page Rank usando α = 1/5. Visualizen la red asignando un tamaño a
cada nodo proporcional al Page Rank que le toca

b. Construyendo la red conectando a cada museo con sus m vecinos más cercanos,
para m = 1, 3, 5, 10 y usando α = 1/5.

c. Para m = 5, considerando los valores de α = 6/7, 4/5, 2/3, 1/2, 1/3, 1/5, 1/7.
"""
K = f.calcula_matrizK(A)
#f.inversa(K)
#Base de función
alpha = 1/5
C = f.calcula_matriz_C(A)
N = A.shape[0]
m_inv = f.inversa(f.calcular_matriz_M(C, N, alpha))
p = m_inv @ np.ones(N)
p.sum() # Verifico que efectivamente sea un versor

fig, ax = plt.subplots(figsize=(10, 10))
barrios.boundary.plot(color='gray',ax=p)
museos.plot(ax=p)


node_sizes =  400 * p  # se puede ajustar tamaños
fig, ax = plt.subplots(figsize=(20, 20))
barrios.to_crs("EPSG:22184").boundary.plot(color='gray', linewidth=0.5, ax=ax)
# Dibuja la red con tamaños proporcionales al PageRank
nx.draw_networkx(
    G,
    pos=G_layout,
    ax=ax,
    node_size=node_sizes,  # escalado
    node_color='purple',      
    edge_color='violet',    
    width=0.5,            
    alpha=0.7,            
    with_labels=False     
)

ax.set_title("Red de Museos en CABA - Tamaño según PageRank (α=1/5)", fontsize=16, pad=20)
ax.grid(False)  # Ocultar cuadrícula

plt.tight_layout()
plt.show()

def visualizar_p(A,alfa,size):
    C = f.calcula_matriz_C(A)
    N=A.shape[0]
    p = f.inversa(f.calcular_matriz_M(C, N, alfa)) @ np.ones(N)

    fig, ax = plt.subplots(figsize=(size['plot_size'], size['plot_size']))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', linewidth=0.5, ax=ax)
    # Dibuja la red con tamaños proporcionales al PageRank
    node_sizes =  size[1] * p  # se puede ajustar tamaños
    nx.draw_networkx(
        G,
        pos=G_layout,
        ax=ax,
        node_size=node_sizes,  # escalado
        node_color='purple',      
        edge_color='violet',    
        width=0.5,            
        alpha=0.7,            
        with_labels=False     
    )
    
    ax.set_title("Red de Museos en CABA - Tamaño según PageRank (α={})".format(size['alpha']), fontsize=16, pad=20)
    ax.grid(False)  # Ocultar cuadrícula
    
    plt.tight_layout()
    plt.show()






#------------------------------------------------------------------------------

factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
pr = np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
pr = pr/pr.sum() # Normalizamos para que sume 1
Nprincipales = 5 # Cantidad de principales
principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres

