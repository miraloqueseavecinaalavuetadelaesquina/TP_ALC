#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 01:31:02 2025

@author: kanxo
"""

import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import LinAlgError

# Función auxiliar para verificar si una matriz es cuadrada
def esMatrizCuadrada(A):
    return A.shape[0] == A.shape[1]

# Función auxiliar para permutar filas (usada en descompPLU)
def permutacion(matriz, vector_P, index):
    n = matriz.shape[0]
    max_index = index + np.argmax(np.abs(matriz[index:, index]))
    if max_index != index:
        matriz[[index, max_index]] = matriz[[max_index, index]]
        vector_P[[index, max_index]] = vector_P[[max_index, index]]

# Descomposición LU sin pivoteo
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

# Ejecutar tests
test_inversa()