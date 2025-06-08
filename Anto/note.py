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
path_t40= '/home/an2/git/tp_alc/TP_ALC'
#path_X270 = '/home/kanxo/git/tp_ALC/Anto/'
sys.path.append(path_t40)

import funciones as f
#import func as ff
#import fun as fff

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
barrios.boundary.plot(color='gray',ax=ax)
museos.plot(ax=ax)

print("Leega hast aca ?")

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

def visualizar_p(A,size,alfa=1/5):
    C = f.calcula_matriz_C(A)
    N=A.shape[0]
    p = f.inversa(f.calcular_matriz_M(C, N, alfa)) @ np.ones(N)

    fig, ax = plt.subplots(figsize=(size['plot size'], size['plot size']))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', linewidth=0.5, ax=ax)
    # Dibuja la red con tamaños proporcionales al PageRank
    node_sizes =  size['node size'] * p  # se puede ajustar tamaños
    nx.draw_networkx(
        G,
        pos=G_layout,
        ax=ax,
        node_size=node_sizes,  # escalado
        node_color='purple',      
        edge_color='blue',    
        width=0.5,            
        alpha=0.7,            
        with_labels=False     
    )
    
    ax.set_title("Red de Museos en CABA - Tamaño según PageRank (α={})".format(alfa), fontsize=16, pad=20)
    ax.grid(False)  # Ocultar cuadrícula
    
    plt.tight_layout()
    plt.show()

def visualizar_pR(D,size,m=3, alfa=1/5):
    A = construye_adyacencia(D,m)

    # Construcción de la red en NetworkX (sólo para las visualizaciones)
    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}

    # Visualizacion
    #fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
    #barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    #nx.draw_networkx(G,G_layout,ax=ax) # Graficamos los museos

    C = f.calcula_matriz_C(A)
    N=A.shape[0]
    p = f.inversa(f.calcular_matriz_M(C, N, alfa)) @ np.ones(N)

    fig, ax = plt.subplots(figsize=(size['plot size'], size['plot size']))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', linewidth=0.5, ax=ax)
    # Dibuja la red con tamaños proporcionales al PageRank
    node_sizes =  size['node size'] * p  # se puede ajustar tamaños
    nx.draw_networkx(
        G,
        pos=G_layout,
        ax=ax,
        node_size=node_sizes,  # escalado
        node_color='purple',      
        edge_color='blue',    
        width=0.5,            
        alpha=0.7,            
        with_labels=False     
    )
    
    ax.set_title("Red de Museos en CABA - Tamaño según PageRank (α={})".format(alfa), fontsize=16, pad=20)
    ax.grid(False)  # Ocultar cuadrícula
    
    plt.tight_layout()
    plt.show()
    
    return p
    

# 3.a
size ={'node size':1000, 'plot size':15}
visualizar_p(A,size)

# 1. Obtener los top 3 museos
top_museos = museos.iloc[p.sort_values(ascending=False).head(3).index].copy()

# 2. Crear el gráfico solo con los museos top
fig, ax = plt.subplots(figsize=(12, 12))

# Graficar barrios (contorno)
barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax, linewidth=0.8)

# Graficar museos top (en rojo, tamaño proporcional al PageRank)
top_museos.to_crs("EPSG:22184").plot(
    ax=ax,
    color='red',
    markersize=top_museos.index.map(p) * 1000,  # Ajusta el escalado (*1000 para visibilidad)
    alpha=0.8,
    edgecolor='black'
)

# Añadir etiquetas con el valor de PageRank
for idx, row in top_museos.iterrows():
    ax.annotate(
        text=f"{p[idx]:.2f}",
        xy=(row.geometry.x, row.geometry.y),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8)
    )

# Configuración adicional
ax.set_title('Top 3 Museos por PageRank', fontsize=14)
ax.set_axis_off()  # Ocultar ejes
plt.show()

# Recopilamos info 
museos_centrales = {'puntajes': [] , 'posiciones': [], 'vectores': []}
n = 3
# 3.b
for m in (1, 3, 5, 10):
    p = visualizar_pR(D,size, m=m)
    p = pd.Series(p).sort_values(ascending=False)
    museos_centrales['vectores'].append(p)
    museos_centrales['puntajes'].append(p.head(n).values.tolist())
    museos_centrales['posiciones'].append(p.head(n).index.tolist())

museos_centrales['puntajes'] = np.array(museos_centrales['puntajes']).transpose()
museos_centrales['posiciones'] = np.array(museos_centrales['posiciones']).transpose()
museos_centrales['titulos'] = ['m=1', 'm=3', 'm=5', 'm=10']


# 3.a'

fig, ax = plt.subplots()
ax.plot([1,3,5,10], museos_centrales['puntajes'][0] , label='Primer museo' ,color='#30BFDE', marker='^', linewidth=2.2, linestyle='-')
ax.plot([1,3,5,10], museos_centrales['puntajes'][1] , label='Segundo museo' ,color='purple', marker='o', linewidth=2.2, linestyle='--')
ax.plot([1,3,5,10], museos_centrales['puntajes'][2] , label='Tercer museo' ,color='violet', marker='o', linewidth=2.2, linestyle='--')

# mostrar titulo
ax.set_title('Puntajes')

# Labels 
ax.set_ylabel('Puntaje PageRank')
ax.set_xlabel('Cantidad de vecinos más cercanos (m)')

# Default grid
ax.grid()
plt.show()
plt.close()


# 3.b'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))   # o simplemente plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5, hspace=0.5) # Con esto indicamos el espacio libre entre los subplots

for ax, p, titulo in zip(axes.flat, museos_centrales['vectores'] , museos_centrales['titulos']):
    # Obtener top 3 museos para este caso
    top_museos = museos.iloc[p.sort_values(ascending=False).head(3).index]
    
    # Graficar barrios (contorno)
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax, linewidth=0.5)
    
    # Graficar museos top (color y tamaño proporcional a p)
    top_museos.to_crs("EPSG:22184").plot(
        ax=ax,
        color='purple',
        markersize=top_museos.index.map(p) * 800,  # Ajusta el escalado
        alpha=0.7
    )
    
    # Añadir etiquetas
    for idx, row in top_museos.iterrows():
        ax.annotate(
            text=f"{p[idx]:.2f}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Configuración del subplot
    ax.set_title(titulo, fontsize=12)
    ax.set_axis_off()  # Ocultar ejes

plt.tight_layout()
plt.show()



# Reiniciamos
museos_centrales = {'puntajes': [] , 'posiciones': [], 'vectores': []}


# 3.c
for alpha in [1/7, 1/5, 1/3, 1/2, 2/3, 4/5, 6/7]:
    p = visualizar_pR(D,size, m=5, alfa=alpha)
    p = pd.Series(p).sort_values(ascending=False)
    museos_centrales['vectores'].append(p)
    museos_centrales['puntajes'].append(p.head(n).values.tolist())
    museos_centrales['posiciones'].append(p.head(n).index.tolist()) 

museos_centrales['puntajes'] = np.array(museos_centrales['puntajes']).transpose()
museos_centrales['posiciones'] = np.array(museos_centrales['posiciones']).transpose()
museos_centrales['titulos'] = ['α=1/7', 'α=1/5', 'α=1/3', 'α=1/2', 'α=2/3', 'α=4/5', 'α=6/7']


# 3.a'

fig, ax = plt.subplots()
ax.plot([1/7, 1/5, 1/3, 1/2, 2/3, 4/5, 6/7], museos_centrales['puntajes'][0] , label='Primer museo' ,color='#30BFDE', marker='^', linewidth=1, linestyle='-')
ax.plot([1/7, 1/5, 1/3, 1/2, 2/3, 4/5, 6/7], museos_centrales['puntajes'][1] , label='Segundo museo' ,color='purple', marker='o', linewidth=1, linestyle='--')
ax.plot([1/7, 1/5, 1/3, 1/2, 2/3, 4/5, 6/7], museos_centrales['puntajes'][2] , label='Tercer museo' ,color='violet', marker='*', linewidth=1, linestyle='-.')

# mostrar titulo
ax.set_title('Puntajes')

# Labels 
ax.set_ylabel('Puntaje PageRank')
ax.set_xlabel('Factor de amortiguamiento (α)')

# Default grid
ax.grid()
plt.show()
plt.close()


# 3.b'

num_plots = len(museos_centrales['titulos'])
nrows = (num_plots + 1) // 2  # Calcula filas necesarias
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 5*nrows))  # Ajusta altura dinámicamente
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# Aplanamos axes y recorremos solo los necesarios
axes = axes.flatten()
for ax, p, titulo in zip(axes[:num_plots], museos_centrales['vectores'], museos_centrales['titulos']):
    # Obtener top 3 museos para este caso
    top_museos = museos.iloc[p.sort_values(ascending=False).head(3).index]
    
    # Graficar barrios (contorno)
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax, linewidth=0.5)
    
    # Graficar museos top (color y tamaño proporcional a p)
    top_museos.to_crs("EPSG:22184").plot(
        ax=ax,
        color='purple',
        markersize=top_museos.index.map(p) * 800,  # Ajusta el escalado
        alpha=0.7
    )
    
    # Añadir etiquetas
    for idx, row in top_museos.iterrows():
        ax.annotate(
            text=f"{p[idx]:.2f}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Configuración del subplot
    ax.set_title(titulo, fontsize=12)
    ax.set_axis_off()  # Ocultar ejes


# Ocultar ejes de subplots no usados
for j in range(num_plots, len(axes)):
    axes[j].set_axis_off()

plt.tight_layout()
plt.show()


"""
Usando la Eq. 5, y suponiendo que las personas dan r = 3 pasos en la red de museos,
calcular la cantidad total de visitantes que entraron en la red, ||v||1 , a partir del vector
w provisto en el archivo visitas.txt. Usar para esto la matriz de transiciones definida
por la Eq. 4. Para esto:
• Construya una función calcula matriz C continua que reciba la matriz de dis-
tancias entre museos D y retorne la matriz C definida en la Eq. 4.

(1/D)^T*(D)

• Construya una función calcula B(C,r) que reciba la matriz C y el número de
pasos r como argumento, y retorne la matriz B de la Eq. 5.
• Utilice la función calculaLU para resolver la Eq. 5.
"""





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

