# -*- coding: utf-8 -*-
"""
Created on Fri May  9 04:55:29 2025

@author: MARCELOFGB
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Configuración para un estilo de gráfico agradable
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Simulación de Datos de Soluciones ---
# Supongamos que un algoritmo evolutivo ha generado un conjunto de posibles "estrategias"
# Cada estrategia tiene un (gasto_administrativo, rendimiento_financiero)

np.random.seed(42) # Para reproducibilidad

# Generamos algunas soluciones que claramente estarán en el frente de Pareto
# (Bajos gastos, bajo/medio rendimiento; Medios gastos, medio/alto rendimiento; Altos gastos, alto rendimiento)
# Estos son nuestros "campeones" teóricos
gastos_pareto_ideal = np.array([20, 25, 30, 40, 55, 75, 90]) # Miles de $
rendimiento_pareto_ideal = np.array([150, 145, 138, 125, 105, 80, 60]) # Miles de $
# Ajuste para que sea estrictamente Pareto: si el gasto aumenta, el rendimiento DEBE disminuir
# para que sea un trade-off real en un frente de MIN gasto, MAX rendimiento.
# O, si el gasto aumenta, el rendimiento AUMENTA, pero con rendimientos decrecientes para el gasto
# Vamos a usar el segundo caso, más intuitivo: para más rendimiento, aceptas más gasto.
# Frente de Pareto: Queremos MINIMIZAR gastos, MAXIMIZAR rendimiento
# Gasto (X), Rendimiento (Y)
# Un punto (g1, r1) domina a (g2, r2) si g1 <= g2 Y r1 >= r2 (y al menos una es estricta)

# Puntos que formarán nuestro frente de Pareto "objetivo"
# Menor gasto implica menor rendimiento. Para mayor rendimiento, se acepta mayor gasto.
# Estos son los "mejores compromisos"
gastos_objetivo_frente = np.array([20,  30,  40,  55,  70,  90, 110]) # Miles de $
rendimiento_objetivo_frente = np.array([60, 80, 95, 105, 110, 112, 113]) # Miles de $

# Generamos soluciones dominadas alrededor de estos puntos del frente
num_soluciones_dominadas_por_punto = 8
gastos_dominados = []
rendimiento_dominados = []

for i in range(len(gastos_objetivo_frente)):
    g_ref = gastos_objetivo_frente[i]
    r_ref = rendimiento_objetivo_frente[i]
    # Generar puntos dominados:
    # 1. Mayor gasto, igual o menor rendimiento
    # 2. Igual gasto, menor rendimiento
    # 3. Mayor gasto, menor rendimiento
    for _ in range(num_soluciones_dominadas_por_punto):
        # Perturbación para gasto (siempre mayor o igual)
        delta_g = np.random.uniform(0, 40)
        # Perturbación para rendimiento (siempre menor o igual)
        delta_r = np.random.uniform(0, 30)

        tipo_dominado = np.random.choice([1,2,3])
        if tipo_dominado == 1: # Mayor gasto, menor rendimiento
            g_new = g_ref + delta_g * (1 if delta_g > 1 else 2) #asegurar que sea mayor
            r_new = r_ref - delta_r * (1 if delta_r > 1 else 2) #asegurar que sea menor
        elif tipo_dominado == 2: # Igual gasto, menor rendimiento
            g_new = g_ref
            r_new = r_ref - delta_r * (1 if delta_r > 1 else 2)
        else: # Mayor gasto, igual rendimiento
            g_new = g_ref + delta_g * (1 if delta_g > 1 else 2)
            r_new = r_ref
        
        # Asegurar que no sean mejores que el punto de referencia
        if g_new >= g_ref and r_new <= r_ref and (g_new > g_ref or r_new < r_ref):
             # Evitar valores negativos o absurdos
            gastos_dominados.append(max(0, g_new))
            rendimiento_dominados.append(max(0, r_new))


gastos_dominados = np.array(gastos_dominados)
rendimiento_dominados = np.array(rendimiento_dominados)

# Combinamos todas las soluciones (las del frente objetivo y las dominadas)
todos_gastos = np.concatenate((gastos_objetivo_frente, gastos_dominados))
todos_rendimientos = np.concatenate((rendimiento_objetivo_frente, rendimiento_dominados))
soluciones = np.array([(g,r) for g,r in zip(todos_gastos, todos_rendimientos)])

# --- 2. Identificación del Frente de Pareto ---
# Una solución 'a' domina a 'b' si:
#   a.gasto <= b.gasto Y a.rendimiento >= b.rendimiento (para Min Gasto, Max Rendimiento)
#   Y (a.gasto < b.gasto O a.rendimiento > b.rendimiento) (al menos una estricta)

indices_frente_pareto = []
for i in range(len(soluciones)):
    es_dominada = False
    for j in range(len(soluciones)):
        if i == j:
            continue
        # Si la solución j domina a la solución i
        if (soluciones[j,0] <= soluciones[i,0] and soluciones[j,1] >= soluciones[i,1]) and \
           (soluciones[j,0] < soluciones[i,0] or soluciones[j,1] > soluciones[i,1]):
            es_dominada = True
            break
    if not es_dominada:
        indices_frente_pareto.append(i)

soluciones_pareto = soluciones[indices_frente_pareto]
# Ordenar el frente de Pareto por gasto para un ploteo más limpio de la línea
soluciones_pareto = soluciones_pareto[soluciones_pareto[:,0].argsort()]

# Separar las soluciones dominadas para el ploteo
indices_dominados = np.array(list(set(range(len(soluciones))) - set(indices_frente_pareto)))
soluciones_dominadas = soluciones[indices_dominados]


# --- 3. Visualización ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plotear todas las soluciones (incluyendo las que serán identificadas como dominadas)
ax.scatter(soluciones_dominadas[:,0], soluciones_dominadas[:,1], 
           c='lightblue', s=50, alpha=0.7, label='Soluciones Dominadas (Subóptimas)')

# Plotear las soluciones del frente de Pareto
ax.scatter(soluciones_pareto[:,0], soluciones_pareto[:,1], 
           c='red', s=70, edgecolor='black', zorder=3, label='Frente de Pareto (Soluciones Óptimas)')

# Línea conectando los puntos del frente de Pareto
ax.plot(soluciones_pareto[:,0], soluciones_pareto[:,1], 
        color='red', linestyle='-', linewidth=2, zorder=2)

# Anotaciones y estilo del gráfico
ax.set_xlabel('Gastos Administrativos (Minimizar →)', fontsize=14)
ax.set_ylabel('Rendimiento Financiero (Maximizar ↑)', fontsize=14)
ax.set_title('Frente de Pareto: Rendimiento Financiero vs. Gastos Administrativos', fontsize=16, pad=20)

# Flechas indicando la dirección de optimización
ax.annotate('', xy=(ax.get_xlim()[0] - 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]), np.median(ax.get_ylim())), # inicio de la flecha de gastos (izquierda)
            xytext=(ax.get_xlim()[0], np.median(ax.get_ylim())), # fin de la flecha
            arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8),
            ha='right', va='center')

ax.annotate('', xy=(np.median(ax.get_xlim()), ax.get_ylim()[1] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0])), # inicio de la flecha de rendimiento (arriba)
            xytext=(np.median(ax.get_xlim()), ax.get_ylim()[1]), # fin de la flecha
            arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8),
            ha='center', va='bottom')

# Formatear los ejes para que muestren "miles de $"
formatter_miles_k = mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k $' if x >=1000 else f'{x:.0f} $')
# Para este ejemplo, los números son pequeños, así que usamos "€" directamente
formatter_euros = mticker.FuncFormatter(lambda x, p: f'{x:.0f}')

ax.xaxis.set_major_formatter(formatter_euros)
ax.yaxis.set_major_formatter(formatter_euros)
ax.tick_params(axis='both', which='major', labelsize=12)


# Añadir anotaciones didácticas sobre puntos específicos del frente
if len(soluciones_pareto) >= 2:
    idx_bajo_gasto = 0 # El primero después de ordenar
    idx_alto_rendimiento = -1 # El último después de ordenar (que tendrá el mayor gasto)
    
    # Estrategia Conservadora (Bajo Gasto, Rendimiento Moderado)
    p_conservador = soluciones_pareto[idx_bajo_gasto]
    ax.annotate(f'Estrategia Conservadora\nBajos Gastos ({p_conservador[0]:.0f} $)\nRendimiento Moderado ({p_conservador[1]:.0f} $)',
                xy=(p_conservador[0], p_conservador[1]),
                xytext=(p_conservador[0] - 15, p_conservador[1] + 15), # Offset del texto
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=0.5, alpha=0.8))

    # Estrategia Agresiva (Alto Gasto, Máximo Rendimiento posible en el frente)
    p_agresivo = soluciones_pareto[idx_alto_rendimiento]
    ax.annotate(f'Estrategia Agresiva\nAltos Gastos ({p_agresivo[0]:.0f} $)\nMáx. Rendimiento ({p_agresivo[1]:.0f} $)',
                xy=(p_agresivo[0], p_agresivo[1]),
                xytext=(p_agresivo[0] - 30, p_agresivo[1] - 20), # Offset del texto
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="black", lw=0.5, alpha=0.8))

    # Punto intermedio si existe
    if len(soluciones_pareto) > 2:
        idx_medio = len(soluciones_pareto) // 2
        p_medio = soluciones_pareto[idx_medio]
        ax.annotate(f'Estrategia Equilibrada\nGastos ({p_medio[0]:.0f} $)\nRendimiento ({p_medio[1]:.0f} $)',
                    xy=(p_medio[0], p_medio[1]),
                    xytext=(p_medio[0] + 10, p_medio[1] + 5), # Offset del texto
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black", lw=0.5, alpha=0.8))


ax.legend(fontsize=12, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()