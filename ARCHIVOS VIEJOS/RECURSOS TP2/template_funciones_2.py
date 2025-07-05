# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])

# =============================================================================
# LIBRERÍAS
# =============================================================================

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

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
    except (ValueError, np.LinAlgError):
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

def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    # definimos el vector s de signos asociado a v
    s = np.array([-1 if x < 0 else (1 if x > 0 else 0) for x in v])
    lambdon = s.T @ L @ s
    return lambdon

#def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
#    return Q

def calcula_Q_(A,R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.array([-1 if x < 0 else (1 if x > 0 else 0) for x in v])
    Q = 1/(2*np.sum(A)) * s.T @ R @ s
    return Q

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

# deflA = A - l v*v^t/(v^tv)
def deflaciona(A,tol=1e-8,maxrep=np.Inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1/(v1.T @ v1) * np.outer(v1,v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.Inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeros autovectores y autovalores de A}
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
   v1,l1,es_conv = metpot1(iX)
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ = metpot1(defliX)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


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

