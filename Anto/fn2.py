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
from numpy.linalg import LinAlgError
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
    k = np.eye(A.shape[0]) @ calcula_K(A)
    R = A-  1/np.sum(A) * k @ np.transpose(A)
    return R

# lambda =(v^t . L . v)/4
def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    lambdon = 1/4 * v.transpose @ L @ v
    return lambdon


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


# mu>0, (L+mu*I)^{-1} 
def metpotI(A,mu,tol=1e-8,maxrep=np.Inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    L = calcula_L(A)
    M = inversa(L + mu * np.identity(A.shape[0]))
    return metpot1(M,tol=tol,maxrep=maxrep)




# =============================================================================
# TEST
# =============================================================================




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

v = np.array([1,2,3])
np.outer(v, v)

np.sum(A_ejemplo)

k = calcula_K(A_ejemplo)

l = calcula_L(A_ejemplo)

r = calcula_R(A_ejemplo)

#v = metpot1(A_ejemplo)
