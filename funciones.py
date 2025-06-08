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
import scipy
import pandas as pd
#from progress.bar import Bar    

# =============================================================================
# IO
# =============================================================================
"""
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
    
"""

# Función para permutar filas (para descompPLU)
def permutacion(A, vector_P, index):
    n = A.shape[0]
    max_index = index + np.argmax(np.abs(A[index:, index]))
    #swap
    if max_index != index:
        A[[index, max_index]] = A[[max_index, index]]
        vector_P[[index, max_index]] = vector_P[[max_index, index]]

"""
# Veersion D&C con error de modelo
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
"""


# Descomposición LU sin pivoteo
def descompLU(A, verbose=False):
    Ac = A.copy().astype(np.float64)
    n = A.shape[0]
    for i in range(n - 1):
        a_ii = Ac[i, i]
        if a_ii == 0:
            raise ValueError("Cero en la diagonal durante LU (se requiere pivoteo)")
        L_i = Ac[i+1:, i] / a_ii
        Ac[i+1:, i] = L_i
        Ac[i+1:, i+1:] -= np.outer(L_i, Ac[i, i+1:])
    
    L = np.tril(Ac, -1) + np.eye(n)
    U = np.triu(Ac)
    if verbose:
        print("L:\n", L)
        print("U:\n", U)
    return L, U

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




# --- Tests ---
def test_inversa():
    # Test 1: Matriz diagonal
    D = np.diag([2, 4, 5])
    D_inv_esperada = np.diag([0.5, 0.25, 0.2])
    D_inv_calculada = inversa(D)
    assert np.allclose(D_inv_calculada, D_inv_esperada), f"Error en matriz diagonal:\n{D_inv_calculada}"
    
    # Test 2: Matriz general invertible
    A = np.array([[1, 2], [3, 4]])
    A_inv_esperada = np.linalg.inv(A)
    A_inv_calculada = inversa(A)
    assert np.allclose(A_inv_calculada, A_inv_esperada), f"Error en matriz general:\n{A_inv_calculada}"
    
    # Test 3: Matriz con pivoteo requerido
    B = np.array([[0, 1], [1, 0]])
    B_inv_esperada = np.linalg.inv(B)
    B_inv_calculada = inversa(B)
    assert np.allclose(B_inv_calculada, B_inv_esperada), f"Error en matriz con pivoteo:\n{B_inv_calculada}"
    
    print("¡Todos los tests pasaron correctamente!")


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

"""  
    
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
"""
#------------------------------------------------------------------------------

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



def calcula_matrizK (A):
    n = A.shape[0]
    k = np.zeros((n,n),dtype=A.dtype)
    for i in range(n):
        k[i][i] = A[i].sum()
    
    return k


#------------------------------------------------------------------------------


def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C    
    Kinv = inversa(calcula_matrizK(A))
    C = np.transpose(A) @ Kinv
    return C

# -----------------------------------------------------------------------------
# M = N/α (I − (1 − α)C)
# falta comprobar la presición de maquina?
def calcular_matriz_M(C,N,alpha):
    ide = np.identity(N)
    M = N/alpha * (ide - (1-alpha)*C)
    return M

 

c = np.array([[0,0,2,4],[0,2,4,3],[5,2,1,1],[2,3,1,1]])
calcular_matriz_M(c, 4, 0.25)
#calculaLU(c)
calcula_matrizK(c)
inversa(calcula_matrizK(c))
#------------------------------------------------------------------------------

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = calcular_matriz_M(C, N, alfa)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = ... # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

#------------------------------------------------------------------------------

# f :: [a] -> [i,a]  
# g :: [i,a] -> [i,a]

def sort_array(v):
    # [a] -> [i,a]
    v_serie = pd.Series(v)
    # ordenamos
    return  v_serie.sort_values()

v = np.array([1,-3,-4,0,0.2,-0.4,3])
v = pd.Series(v).sort_values(ascending=False)
v.iat[0]
w = v.head(3).index
type(v.head(3).index.to_numpy())
v.head(3).values
#------------------------------------------------------------------------------
# La diagonal son ceros
def calcula_matriz_C_continua_(D):
    N = D.shape[0]
    C = np.eye(N) # acá C es la matriz identidad
    M =D + C
    M = 1/M
    M -= C
    # De paso sumemos las columnas
    suma_columnas = np.transpose(np.add.reduce(M,axis=0))
    #y de paso multipliquuemos 
    for i in range(N):
        M[i] = M[i] * suma_columnas
    return np.transpose(M)

A = np.array([[0,1,2,3],[4,0,3,4],[2,2,0,12],[4,1,8,0]])
vector_a = np.array([1, 2])
vector_b = np.array([0.5, 0.25])
vector_a * vector_b
m = calcula_matriz_C_continua_(A)

def norma2(v, l, r):
    if l+1 < r:
        mid = (l+r)//2
        n_left = norma2(v, l, mid)
        n_right = norma2(v, mid, r)
        return n_left + n_right
    else:
        return v[l]*v[l]

def norma2Vectorial(v):
    vc = v.copy()
    norma = norma2(vc,l=0,r=v.shape[0])    
    return np.sqrt(norma)

#------------------------------------------------------------------------------


def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = np.zeros(D.shape, dtype=D.dtype) # Matriz de transiciones
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i][j] != 0:
                F[i][j] = 1 / D[i][j]
    np.fill_diagonal(F,0)
    Kinv = inversa(calcula_matrizK(D)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = Kinv @ F # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        B = B + np.linalg.matrix_power(C, i) # Sumamos las matrices de transición para cada cantidad de pasos
    return B



