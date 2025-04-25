#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""
Idea general de la descomposicion A = LU.
particion de matrices
"""

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def esMatrizCuadrada(A):
    return A.shape[0] == A.shape[1]

def matriz_H(n, p=np.float64 ):
    H = np.zeros((n,n),dtype=p)
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/(i+j+1)
    return H


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


# Information gain in B1 and B2, B1 or B2
def calcularNormaF(A,i_up, i_dw, j_l, j_r):
    if i_up+1 < i_dw or j_l+1 < j_r:
        i_mid = (i_up + i_dw)//2
        j_mid = (j_l + j_r)//2
        if j_l+1 < j_r:           
            n11 = calcularNormaF(A, i_up, i_mid, j_l, j_mid)
            n12 = calcularNormaF(A, i_up, i_mid, j_mid, j_r)
            n21 = calcularNormaF(A, i_mid, i_dw, j_l, j_mid)
            n22 = calcularNormaF(A, i_mid, i_dw, j_mid, j_r)
            return n11 + n12 + n21 + n22
        elif i_up+1 < i_dw:
            n11 = calcularNormaF(A, i_up, i_mid, j_l, j_mid)
            n12 = calcularNormaF(A, i_up, i_mid, j_mid, j_r)
            return n11 + n12
        else:
            n11 = calcularNormaF(A, i_up, i_mid, j_l, j_mid)
            n12 = calcularNormaF(A, i_up, i_mid, j_mid, j_r)
            return n21 + n22        
    else:
        return A[i_up][j_l]*A[i_up][j_l]

def normaFrobenius(A):
    n, m = A.shape[0], A.shape[1]
    norma = calcularNormaF(A,0,n,0,m)
    print(norma)
    return  np.sqrt(norma)

def normaF2(A):
    suma_cuadrados = np.sum(np.square(A))
    normaF = np.sqrt(suma_cuadrados)
    return normaF



#función para generar matrices dada una condicion externa
def generarMatriz(fil=2, col=2, metodo='random', radio=1, p=np.float64):
    if metodo == 'random':
        return (2*radio * np.random.rand(fil,col) - radio).astype(p)
    elif metodo == 'H' and fil == col:
        return matriz_H(fil,p=p)
    

def errorRelativo(M,v_x, v_xp, v_b, funcion_norma,  condicion=True):
    if condicion:
        n = funcion_norma(M@v_xp-v_b)
        d = funcion_norma(v_b)
    else:
        n = funcion_norma(v_xp-v_x)
        d = funcion_norma(v_x)
    if n == 0 or d == 0:
        return 0
    else:
        return np.log(n/d)


"""
k, auxk, n = index, index, A.shape[0]
# Buscamos los indices para hacer el swap de filas
while k < n and A[index][k] == 0:
    if A[k][index] != 0:
        auxk = k
    k+=1
# swapiamos
fila_i = np.array(A[index])
if  auxk != index and A[k][index] == 0:
    k = auxk
else:
    print("Fallo!, error en matriz")
    return
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

def acotarK(inf, sup, k):
    if inf > k:
        inf = k
    elif sup < k:
        sup = k
    

"""
función auxiliar para ejercicio 7,8,9
Aviso: está función tiene un comportamiento defectuoso, hay que corregir.
warning:
en la sumatoria de los errores relativos, si el error es muy pequeño, entonces sum(e_i) -> 0, entonces log(sum(e_i)) -> -inf
posible solución:   crear una funcion de suma que tenga en cuenta el error de maquina (idem labo 2), tener en cuenta la mantisa donde b = 2 y m = 52
                    cual es la mejor manera de aproximar la aproximacion hecha por la maquina 
                    
                    assert tipo_num == A.dtype and tipo_num == x.dtype and tipo_num ==b.dtype and tipo_num == A_i.dtype and tipo_num == xp.dtype , "Error de tipos"
"""

def perfomance(dict_func, rango_de_iteracion=(2,10), ciclos=10, tipo_de_matriz='random', tipo_num=np.float64 ,er_b=True, verbose=True):
    v1 = np.zeros(rango_de_iteracion[1]+1, dtype=tipo_num)
    v2 = np.zeros(rango_de_iteracion[1]+1, dtype=tipo_num)
    if verbose: bar = Bar('Procesando', max=rango_de_iteracion[1]+1, suffix='%(percent)d%%')    
    for i in range(rango_de_iteracion[0],rango_de_iteracion[1]+1):
        e1,e2 = tipo_num(0), tipo_num(0)
        for j in range(ciclos):
            A = generarMatriz(fil=i,col=i,metodo=tipo_de_matriz, p=tipo_num)
            x = (2 * np.random.rand(i,1) - 1).astype(tipo_num)
            b = A@x
            A_i = dict_func['inv'](A)
            xp = dict_func['solve'](A,b)
            e = errorRelativo(M=A, v_x=x, v_xp=xp, v_b=b, funcion_norma=dict_func['norma'])
            e1 += e 
            xp = A_i@b
            e = errorRelativo(M=A, v_x=x, v_xp=xp, v_b=b, funcion_norma=dict_func['norma'], condicion=False)
            e2 += e
        
        v1[i] = e1
        v2[i] = e2
        #print(i)
        if verbose: bar.next()
    if verbose:
        bar.finish()
        print()
        print("Finalizó con exito")
              
    return v1, v2


def plotearVectores(dicc_v1, dicc_v2, rotulos, start=10):
    if len(dicc_v1['vector']) != len(dicc_v2['vector']):
        print("Parametros invalidos")
        return 
    n = np.arange(start,len(dicc_v1['vector']))
    plt.plot(n,dicc_v1['vector'][start:], label=dicc_v1['leyenda'])
    plt.plot(n,dicc_v2['vector'][start:], label=dicc_v2['leyenda'])
    plt.xlabel(rotulos['eje x'])
    plt.ylabel(rotulos['eje y'])
    plt.title(rotulos['titulo'])
    plt.legend()
    plt.show()
    plt.close()


def plotDistanciaVectores(v1,v2, rotulos, start=10):
    if len(v1) != len(v2):
        print("Parametros invalidos")
        return
    distancia = np.abs(v1 - v2)
    n = np.arange(start,len(v1))
    plt.plot(n,distancia[start:])
    plt.xlabel(rotulos['eje x'])
    plt.ylabel(rotulos['eje y'])
    plt.title(rotulos['titulo'])
    plt.show()
    plt.close()
    
    

A = np.array([1,2])
B = np.array([3,4])
C = np.abs(A-B)
    
# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================
    
# PRE: A es matriz cuadrada
def descompLU(A, verbose=False):
    Ac = A.copy()
    n = A.shape[0]-1
    i = 0
    # Descomponemos y operamos
    while i<n:
        a_11 = Ac[i][i]
        if a_11 == 0: 
            print("cero en diagonal de matriz")
            return Ac,A
            break
        U_12 = Ac[i:i+1,i+1:]        
        L_21 = np.divide(Ac[i+1:,i:i+1],a_11)
        Ac[i+1:,i:i+1] = L_21
        Ac_i = np.subtract( Ac[i+1:,i+1:], L_21@U_12)
        Ac[i+1:,i+1:] = Ac_i
        i+=1
        
    L = np.tril(Ac,-1) + np.eye(A.shape[0], dtype=A.dtype) 
    U = np.triu(Ac)
    if verbose:
        print("Matriz L \n", L)
        print("Matriz U \n", U)        
    return L,U

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

# LUx = b => Ly=b, Ux=y
def resolverLU(A, b):
    n = A.shape[0]
    if not esMatrizCuadrada(A) or b.shape[0] != n:
        print("Matriz y/o vector no válidos")
        return 0    
    L,U = descompLU(A, verbose=False)
    y = solve_triangular(L, b, lower=True).astype(A.dtype) # Ly = b
    x = solve_triangular(U, y).astype(A.dtype) # Ux = y
    return x

def resolverPLU(A, b):
    n = A.shape[0]
    if not esMatrizCuadrada(A) or b.shape[0] != n:
        print("Matriz y/o vector no válidos")
        return 0    
    P,L,U = descompPLU(A, verbose=False)
    y = solve_triangular(L, b, lower=True).astype(A.dtype) # Ly = b
    x = solve_triangular(U, y).astype(A.dtype) # Ux = y
    return x

A = np.array([[0,0,2,4],[0,2,4,3],[5,2,1,1],[2,3,1,1]]) 
P,L,U = descompPLU(A)
A = np.array([[5,2,1,1],[0,2,4,3],[0,0,2,4],[2,3,1,1]]) 
B,C = descompLU(A)
print('B=L? ' , 'Si!' if np.allclose(np.linalg.norm(B - L, 1), 0) else 'No!')
print('C=U? ' , 'Si!' if np.allclose(np.linalg.norm(C - U, 1), 0) else 'No!')


def inversa (A) :
    n = A.shape[0]
    i = 0
    try: 
        L,U = descompLU(A)
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

funciones = {'inv' : inversa,
             'solve' : resolverPLU,
             'norma' : norma2Vectorial}
#v1,v2 = perfomance(funciones,rango_de_iteracion=(10,200), tipo_num=np.float16)

# =============================================================================
# TESTING
# =============================================================================


def testeo(test, tipo_de_dato='np array', autocontenido=False):
    if test['cant_args'] == 1:
        res = test['func'](test['arg1'])
    elif test['cant_args'] == 2:
        res = test['func'](test['arg1'], test['arg2'])
    elif test['cant_args'] == 3:
        res = test['func'](test['arg1'], test['arg2'], test['arg3'])
    else:
        print("ERROR: la función testear() solo soporta funciones de hasta 4 argumentos.")
        return False
    if autocontenido : test['res_correcto'] = test['arg1']
    if tipo_de_dato == 'np array': 
        if test['name'] in {'A=LU', 'PA=LU'}: 
            res = res[0]@res[1]
        print("El Test "+test['name'], 
                  ' está OK!, dio el resultado esperado \n' if np.allclose(np.linalg.norm(test['res_correcto'] - res, 1), 0) else 'ERROR!, falló con resultado \n', res)
    else:    
        if res != test['res_correcto']:
            print("ERROR: el test "+ test['name'] +" falló con resultado " + str(res))
            return False
        else:
            print("OK: el test "+ test['name'] +" dio el resultado esperado.")
            return True



A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])
L = np.array([[1,0,0,0],[2,1,0,0],[-1,3,1,0],[2,-1,3,1]])
print(np.allclose(np.linalg.norm(A - A, 1), 0))
U = np.array([[2,1,2,3],[0,1,-1,-2],[0,0,1,-3],[0,0,0,-2]])
A_i= np.array([[-193/2,39,-12,7/2],[135/2,-27,17/2,-5/2],[89/2,-18,11/2,-3/2],[25/2,-5,3/2,-1/2]])
b = np.array([[8],[14],[-16],[10]])
x = np.array([[1],[1],[1],[1]])

# Test decomposicion LU
test = {'name' : 'A=LU',
        'cant_args' : 1,
        'arg1' : A,
        'res_correcto' : (L,U),
        'func' : descompLU
        }

testeo(test, autocontenido=True)

# Test resolverLU
test = {'name' : 'Ax=b, A=LU',
        'cant_args' : 2,
        'arg1' : A,
        'arg2' : b,
        'res_correcto' : x,
        'func' : resolverLU
        }

testeo(test)

# Test inversa
test = {'name' : 'A*A⁻1 = I',
        'cant_args' : 1,
        'arg1' : A,
        'res_correcto' : A_i,
        'func' : inversa
        }

testeo(test)

A = np.array([[1,0,1],[0,-1,2],[1,1,0]])
L = np.array([[1,0,0],[0,1,0],[1,-1,1]])
U = np.array([[1,0,1],[0,-1,2],[0,0,1]])
A_i = np.array([[2,-1,-1],[-2,1,2],[-1,1,1]])
n = normaFrobenius(A)


    

test['arg1'] = A
test['res_correcto'] = A_i
testeo(test)

test = {'name' : 'A=LU',
        'cant_args' : 1,
        'arg1' : A,
        'res_correcto' : (L,U),
        'func' : descompLU
        }

testeo(test, autocontenido=True)

A= np.array([[1,2,3], [4,1,1]])
B = np.array([[1,3,2],[4,5,1],[0,7,2]])
C = np.array([[2,1], [7, -3]])


#n1=normaFrobenius(A)
for i in (A,B,C):
    n = normaFrobenius(i)
    print(n)