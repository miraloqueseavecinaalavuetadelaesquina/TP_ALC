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


# =============================================================================
# FUNCIONES PRELIMINARES
# =============================================================================

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
    lambdon = 1/4 * np.transpose(v) @ L @ v
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
   v = ... # Generamos un vector de partida aleatorio, entre -1 y 1
   v = ... # Lo normalizamos
   v1 = ... # Aplicamos la matriz una vez
   v1 = ... # normalizamos
   l = ... # Calculamos el autovector estimado
   l1 = ... # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = ... # Calculo nuevo v1
      v1 = ... # Normalizo
      l1 = ... # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = ... # Calculamos el autovalor
   return v1,l,nrep<maxrep


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

np.sum(A_ejemplo)

k = calcula_K(A_ejemplo)

l = calcula_L(A_ejemplo)

r = calcula_R(A_ejemplo)
