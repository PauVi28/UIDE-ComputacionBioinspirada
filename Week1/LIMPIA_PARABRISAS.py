# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 00:07:25 2025

@author: MARCELOFGB
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definir el Universo de Discurso
intensidad_lluvia = ctrl.Antecedent(np.arange(0, 101, 1), 'intensidad_lluvia')
velocidad_limpiaparabrisas = ctrl.Consequent(np.arange(0, 101, 1), 'velocidad_limpiaparabrisas')

# 2. Definir las Funciones de Pertenencia

# Entrada: Intensidad_Lluvia (Gaussianas y Sigmoide)
intensidad_lluvia['no_lluvia'] = fuzz.gaussmf(intensidad_lluvia.universe, 0, 7)
intensidad_lluvia['lluvia_ligera'] = fuzz.gaussmf(intensidad_lluvia.universe, 25, 8)
intensidad_lluvia['lluvia_moderada'] = fuzz.gaussmf(intensidad_lluvia.universe, 55, 10)
intensidad_lluvia['lluvia_intensa'] = fuzz.sigmf(intensidad_lluvia.universe, 75, 0.2) 

# Salida: Velocidad_Limpiaparabrisas (Gaussianas)
velocidad_limpiaparabrisas['apagado'] = fuzz.gaussmf(velocidad_limpiaparabrisas.universe, 0, 7)
velocidad_limpiaparabrisas['intermitente_lento'] = fuzz.gaussmf(velocidad_limpiaparabrisas.universe, 25, 8)
velocidad_limpiaparabrisas['intermitente_rapido'] = fuzz.gaussmf(velocidad_limpiaparabrisas.universe, 50, 10)
velocidad_limpiaparabrisas['continuo_lento'] = fuzz.gaussmf(velocidad_limpiaparabrisas.universe, 75, 8)
velocidad_limpiaparabrisas['continuo_rapido'] = fuzz.gaussmf(velocidad_limpiaparabrisas.universe, 100, 7)

# 3. Definir las Reglas Difusas
rule1 = ctrl.Rule(intensidad_lluvia['no_lluvia'], velocidad_limpiaparabrisas['apagado'])
rule2 = ctrl.Rule(intensidad_lluvia['lluvia_ligera'], velocidad_limpiaparabrisas['intermitente_lento'])
rule3 = ctrl.Rule(intensidad_lluvia['lluvia_moderada'], velocidad_limpiaparabrisas['intermitente_rapido'])
rule4 = ctrl.Rule(intensidad_lluvia['lluvia_intensa'], velocidad_limpiaparabrisas['continuo_rapido'])

# 4. Crear el Sistema de Control Difuso
sistema_control_lluvia = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
simulador_lluvia = ctrl.ControlSystemSimulation(sistema_control_lluvia)

# Gráficos de las Funciones de Pertenencia


intensidad_lluvia.view()
plt.title('Funciones de Pertenencia - Intensidad de Lluvia (Entrada)')
plt.grid(True)


velocidad_limpiaparabrisas.view()
plt.title('Funciones de Pertenencia - Velocidad de Limpiaparabrisas (Salida)')
plt.grid(True)


plt.show()

print("\n--- GRÁFICOS DE FUNCIONES DE PERTENENCIA MOSTRADOS ---")


#### Ejemplo 1: Lluvia Muy Ligera

# Corregido: Guardar el valor de entrada en una variable
input_intensidad_ej1 = 15 

print(f"\n--- EJEMPLO 1: Lluvia Muy Ligera (Intensidad de Lluvia = {input_intensidad_ej1}) ---")
simulador_lluvia.input['intensidad_lluvia'] = input_intensidad_ej1 # Asignar el valor
simulador_lluvia.compute()

velocidad_ej1 = simulador_lluvia.output['velocidad_limpiaparabrisas']
print(f"Velocidad de Limpiaparabrisas calculada: {velocidad_ej1:.2f}%")

# Gráfico de Desfusificación para el Ejemplo 1
velocidad_limpiaparabrisas.view(sim=simulador_lluvia)
plt.title(f'Desfusificación para Intensidad de Lluvia = {input_intensidad_ej1} (Velocidad: {velocidad_ej1:.2f}%)')
plt.grid(True)
plt.show()

# Descripción de la activación de MFs para este ejemplo:
print(f"\nGrado de Membresía de la Entrada (Intensidad de Lluvia = {input_intensidad_ej1}):")
for term, mf_obj in intensidad_lluvia.terms.items(): # CAMBIADO: Renombrado mf a mf_obj para claridad
    # CORRECCIÓN: Acceder a mf_obj.mf para obtener el array de la función de pertenencia
    membership = fuzz.interp_membership(intensidad_lluvia.universe, mf_obj.mf, input_intensidad_ej1) 
    print(f"  {term}: {membership:.2f}")


#### Ejemplo 2: Lluvia Moderada a Intensa

# Corregido: Guardar el valor de entrada en una variable
input_intensidad_ej2 = 70 

print(f"\n--- EJEMPLO 2: Lluvia Moderada a Intensa (Intensidad de Lluvia = {input_intensidad_ej2}) ---")
simulador_lluvia.input['intensidad_lluvia'] = input_intensidad_ej2 # Asignar el valor
simulador_lluvia.compute()

velocidad_ej2 = simulador_lluvia.output['velocidad_limpiaparabrisas']
print(f"Velocidad de Limpiaparabrisas calculada: {velocidad_ej2:.2f}%")

# Gráfico de Desfusificación para el Ejemplo 2
print(f"\nGráfico de Desfusificación para el Ejemplo 2 (Intensidad de Lluvia = {input_intensidad_ej2}):")
velocidad_limpiaparabrisas.view(sim=simulador_lluvia)
plt.title(f'Desfusificación para Intensidad de Lluvia = {input_intensidad_ej2} (Velocidad: {velocidad_ej2:.2f}%)')
plt.grid(True)
plt.show()

# Descripción de la activación de MFs para este ejemplo:
print(f"\nGrado de Membresía de la Entrada (Intensidad de Lluvia = {input_intensidad_ej2}):")
for term, mf_obj in intensidad_lluvia.terms.items(): # CAMBIADO: Renombrado mf a mf_obj para claridad
    # CORRECCIÓN: Acceder a mf_obj.mf para obtener el array de la función de pertenencia
    membership = fuzz.interp_membership(intensidad_lluvia.universe, mf_obj.mf, input_intensidad_ej2) 
    print(f"  {term}: {membership:.2f}")