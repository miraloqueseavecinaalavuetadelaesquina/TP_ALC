"""
TP 1
Grupo ALV
"""
# =============================================================================
# LIBRERÍAS
# =============================================================================

import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from progress.bar import Bar    

# =============================================================================
# IO
# =============================================================================

def permutacion(A, vector_P, index=0):
    # dada una columna col_k(A), buscamos el indice i del valor absoluto mas alto de dicha columna, luego hacemos swap
    k = np.argmax(np.abs(A[index:,index:index+1]))
    # swapiamos
    fila_i = np.array(A[index])
    if  k == 0:
        print("Fallo!, error en matriz")
        return
    k += index
    A[index] = A[k]
    A[k] = fila_i
    vector_P[index], vector_P[k] = vector_P[k], vector_P[index]

def descompPLU(A, verbose=False):
    Ac = A.copy()
    n = A.shape[0]-1
    i = 0
    v_p = np.arange(n+1)
    # Descomponemos y operamos
    while i<n:        
        if Ac[i][i] == 0: 
            permutacion(Ac, vector_P=v_p, index=i)
        a_11 = Ac[i][i]
        U_12 = Ac[i:i+1,i+1:]        
        L_21 = np.divide(Ac[i+1:,i:i+1],a_11)
        Ac[i+1:,i:i+1] = L_21
        Ac_i = np.subtract( Ac[i+1:,i+1:], L_21@U_12)
        Ac[i+1:,i+1:] = Ac_i
        i+=1
    P = np.eye(n+1, dtype=A.dtype)
    L = np.tril(Ac,-1) + P 
    U = np.triu(Ac)
    P = P[v_p]
    if verbose:
        print("Matriz P \n", P)
        print("Matriz L \n", L)
        print("Matriz U \n", U)        
    return P,L,U

# =============================================================================
# T
# =============================================================================



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

def calculaLU(matriz,verbose=False):
    matriz_c = matriz.copy()
    n = matriz.shape[0]-1
    i = 0
    # Descomponemos y operamos
    while i<n:
        a_11 = matriz_c[i][i]
        if a_11 == 0: 
            print("cero en diagonal de matriz")
            return matriz_c,matriz
            break
        U_12 = matriz_c[i:i+1,i+1:]        
        L_21 = np.divide(matriz_c[i+1:,i:i+1],a_11)
        matriz_c[i+1:,i:i+1] = L_21
        matriz_c_i = np.subtract( matriz_c[i+1:,i+1:], L_21@U_12)
        matriz_c[i+1:,i+1:] = matriz_c_i
        i+=1
        
    l = np.tril(matriz_c,-1) + np.eye(matriz.shape[0], dtype=matriz.dtype) 
    u = np.triu(matriz_c)
    if verbose:
        print("Matriz L \n", l)
        print("Matriz U \n", u)        
    return l,u

#------------------------------------------------------------------------------------------

def inversa (A) :
    n = A.shape[0]
    i = 0
    try: 
        L,U = calculaLU(A)
    except ValueError:
        P,L,U = descompPLU(A)
    if np.allclose(np.linalg.norm(A - U, 1), 0): # si descompLU no funciona, probamos con descompPLU
        P,L,U = descompPLU(A)
    inversa = []
    try:
        while i < n:        
            e_i = np.zeros(n)
            e_i[i] = 1
            y = solve_triangular(L, e_i, lower=True).astype(A.dtype)
            x = solve_triangular(U, y, lower=False).astype(A.dtype)
            inversa.append(x)
            i+=1
    except LinAlgError as e:
        print("Error de algebra lineal : ", e.args[0])
    matriz = np.array(inversa)    
    inversa = np.transpose(matriz)
    return inversa


def calcula_matrizK (A):
    n = A.shape[0]
    k = np.zeros((n,n),dtype=A.dtype)
    for i in range(n):
        k[i][i] = A[i].sum()
    
    return k

#-----------------------------------------------------------------------------------------


def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    Kinv = inversa(calcula_matrizK(A))
    C = Kinv @ A
    return C

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = ...
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = ... # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        print(1)
        # Sumamos las matrices de transición para cada cantidad de pasos
    return



