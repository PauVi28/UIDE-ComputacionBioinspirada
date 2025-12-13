# -*- coding: utf-8 -*-
"""
Created on Fri May  9 05:13:00 2025

@author: MARCELOFGB
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 1. Definición del Problema
class ProblemaFinanciero(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,  # x1: NIT, x2: GES
                         n_obj=2,  # RF, GA
                         n_constr=0, # Sin restricciones adicionales por ahora
                         xl=np.array([0, 0]), # Límites inferiores para x1, x2
                         xu=np.array([10, 10]))# Límites superiores para x1, x2

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2 = x[0], x[1] # Desempaquetamos las variables de decisión

        # Objetivo 1: Maximizar Rendimiento Financiero (RF)
        # Como NSGA-II minimiza, debemos pasar -RF
        rf = (x1 + 1)**2 + 0.5*x2 - 0.1*x2**2
        obj1 = -rf # Minimizar -RF es maximizar RF

        # Objetivo 2: Minimizar Gastos Administrativos (GA)
        ga = 50 / (x2 + 1) + 0.2*(x1 - 5)**2 + 5
        obj2 = ga

        out["F"] = [obj1, obj2]

# Instanciar el problema
problem = ProblemaFinanciero()

# 2. Configuración del Algoritmo NSGA-II
algorithm = NSGA2(
    pop_size=100,     # Tamaño de la población
    eliminate_duplicates=True # Buena práctica
)

# 3. Ejecución de la Optimización
res = minimize(problem,
               algorithm,
               ('n_gen', 200), # Número de generaciones
               seed=1,         # Para reproducibilidad
               verbose=True)   # Muestra el progreso

# 4. Extracción y Preparación de Resultados para Graficar
# res.F contiene los valores de los objetivos para las soluciones del frente de Pareto
# Recordar que el primer objetivo (RF) está negado
pareto_rf = -res.F[:, 0]  # Rendimiento Financiero (original, maximizar)
pareto_ga = res.F[:, 1]   # Gastos Administrativos (original, minimizar)

# También podemos obtener las variables de decisión (x1, x2) para cada solución del frente
pareto_x1 = res.X[:, 0]
pareto_x2 = res.X[:, 1]

# 5. Visualización del Frente de Pareto
plt.figure(figsize=(10, 7))
plt.scatter(pareto_ga, pareto_rf, s=50, facecolors='blue', edgecolors='blue', alpha=0.7)
plt.title('Frente de Pareto: Rendimiento Financiero vs. Gastos Administrativos')
plt.xlabel('Gastos Administrativos (GA) - Minimizar')
plt.ylabel('Rendimiento Financiero (RF) - Maximizar')
plt.grid(True)
plt.gca().invert_xaxis() # Opcional: A veces es intuitivo tener "mejor" hacia la derecha para minimizar
                         # Pero el estándar es menor a la izquierda, mayor a la derecha.
                         # Para GA (minimizar), menor es mejor, así que lo dejamos estándar.
                         # RF (maximizar), mayor es mejor, así que estándar está bien.
plt.show()

print("\nAlgunas soluciones del Frente de Pareto (x1: NIT, x2: GES, RF, GA):")
for i in range(min(10, len(pareto_rf))): # Mostramos hasta 10 soluciones
    print(f"Sol {i+1}: NIT={pareto_x1[i]:.2f}, GES={pareto_x2[i]:.2f} -> RF={pareto_rf[i]:.2f}, GA={pareto_ga[i]:.2f}")

# Opcional: Visualización en el espacio de decisión
# Esto puede ayudar a entender qué combinaciones de x1 y x2 llevan al frente.
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pareto_x1, pareto_x2, c=pareto_rf, cmap='viridis', s=50, alpha=0.7)
plt.title('Soluciones del Frente de Pareto en el Espacio de Decisión (NIT vs GES)')
plt.xlabel('Nivel de Inversión en Nuevas Tecnologías (NIT)')
plt.ylabel('Grado de Externalización de Servicios (GES)')
cbar = plt.colorbar(scatter, label='Rendimiento Financiero (RF)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 7))
scatter = plt.scatter(pareto_x1, pareto_x2, c=pareto_ga, cmap='plasma_r', s=50, alpha=0.7) # _r invierte el colormap
plt.title('Soluciones del Frente de Pareto en el Espacio de Decisión (NIT vs GES)')
plt.xlabel('Nivel de Inversión en Nuevas Tecnologías (NIT)')
plt.ylabel('Grado de Externalización de Servicios (GES)')
cbar = plt.colorbar(scatter, label='Gastos Administrativos (GA)')
plt.grid(True)
plt.show()