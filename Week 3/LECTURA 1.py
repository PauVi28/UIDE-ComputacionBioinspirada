# -*- coding: utf-8 -*-
"""
Created on Fri May  9 04:21:18 2025

@author: MARCELOFGB
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm # Para una barra de progreso

# --- Parámetros de la Programación Evolutiva ---
N_INDIVIDUOS = 30        # Tamaño de la población
N_GENERACIONES = 50      # Número de generaciones
PROB_MUTACION = 0.02     # Probabilidad de mutación por gen
TAMANO_TORNEO = 3        # Número de individuos en cada torneo
ALPHA_PENALIZACION = 0.01 # Factor de penalización por tamaño del subset

# --- 0. Preparación de Datos ---
# Usaremos el dataset de cáncer de mama de sklearn
data = load_breast_cancer()
X, y = data.data, data.target

# Dividimos en entrenamiento (para PE) y prueba (para evaluación final)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Dividimos el conjunto de entrenamiento completo en entrenamiento (para el modelo) y validación (para la aptitud)
# Esta división se usará DENTRO de la función de aptitud
# Para simplificar, podríamos usar una porción fija, o mejor, hacer un split cada vez.
# Por ahora, usaremos X_train_full directamente y una sub-división interna para validación en la aptitud.

# Escalar los datos es importante para modelos como Regresión Logística
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

N_INSTANCIAS_TOTALES = X_train_full_scaled.shape[0]

# --- 1. Inicialización ---
def inicializar_poblacion(n_individuos, n_instancias):
    """Genera una población inicial de N individuos aleatoriamente."""
    poblacion = []
    for _ in range(n_individuos):
        # Cada individuo es un vector binario
        # Inicializamos con una probabilidad ~50% de seleccionar cada instancia
        individuo = np.random.randint(0, 2, size=n_instancias)
        # Asegurarse de que al menos una instancia sea seleccionada para evitar errores
        if np.sum(individuo) == 0:
            idx_random = np.random.randint(0, n_instancias)
            individuo[idx_random] = 1
        poblacion.append(individuo)
    return poblacion

# --- 2. Evaluación (Función de Aptitud) ---
def calcular_aptitud(individuo, X_data, y_data, X_val_global, y_val_global, alpha):
    """
    Calcula la aptitud de un individuo.
    Entrena un modelo con las instancias seleccionadas y evalúa en un conjunto de validación.
    """
    indices_seleccionados = np.where(individuo == 1)[0]

    if len(indices_seleccionados) == 0:
        return -1.0 # Penalización máxima si no se selecciona ninguna instancia

    X_subset = X_data[indices_seleccionados]
    y_subset = y_data[indices_seleccionados]

    # Para la validación interna de la aptitud, podemos usar una fracción del X_train_full
    # O, idealmente, tener un X_val, y_val separados de X_train_full
    # Para este ejemplo, usaremos X_val_global, y_val_global que pasamos como argumento
    # (estos serían una parte de X_train_full_scaled, y_train_full reservados para esto)

    # Dividimos X_train_full y y_train_full para tener un set de validación interno para la aptitud
    # Esto es una simplificación. Una K-Fold CV interna sería más robusta pero más lenta.
    if X_subset.shape[0] < 5: # Necesitamos suficientes muestras para entrenar y validar
        return -0.5 # Penalización si el subset es demasiado pequeño

    # Usamos una porción del dataset de entrenamiento original como validación para la aptitud
    # Nota: Sería mejor tener un X_val, y_val fijo para esto, separado del X_train_full
    # O usar K-Fold Cross-Validation dentro de la función de aptitud.
    # Por simplicidad, dividimos X_data (que es X_train_full_scaled) para esta evaluación.
    try:
        # Creamos un split temporal para la evaluación de este individuo
        # Esto asegura que el modelo se entrena y evalúa en datos diferentes dentro de la aptitud
        X_train_apt, X_val_apt, y_train_apt, y_val_apt = train_test_split(
            X_subset, y_subset, test_size=0.3, random_state=1, stratify=y_subset if np.unique(y_subset).size > 1 else None
        )
        if X_train_apt.shape[0] == 0 or X_val_apt.shape[0] == 0: # No se pueden crear splits
            return -0.4

        modelo = LogisticRegression(solver='liblinear', random_state=42, max_iter=100) # max_iter para evitar warnings de convergencia
        modelo.fit(X_train_apt, y_train_apt)
        predicciones = modelo.predict(X_val_apt)
        precision = accuracy_score(y_val_apt, predicciones)
    except ValueError: # Por ejemplo, si solo hay una clase en y_subset_train
        precision = 0.0


    ratio_seleccionadas = len(indices_seleccionados) / len(individuo)
    aptitud = precision - (alpha * ratio_seleccionadas)
    return aptitud

# --- 3. Selección ---
def seleccion_por_torneo(poblacion, aptitudes, tamano_torneo):
    """Selecciona un padre mediante torneo."""
    seleccionados = []
    for _ in range(len(poblacion)): # Seleccionamos N padres
        idx_participantes = np.random.choice(len(poblacion), tamano_torneo, replace=False)
        participantes_aptitud = [aptitudes[i] for i in idx_participantes]
        mejor_idx_local = np.argmax(participantes_aptitud)
        seleccionados.append(poblacion[idx_participantes[mejor_idx_local]])
    return seleccionados

# --- 4. Mutación ---
def mutar(individuo, prob_mutacion):
    """Aplica mutación a un individuo (voltea bits)."""
    descendiente = individuo.copy()
    for i in range(len(descendiente)):
        if np.random.rand() < prob_mutacion:
            descendiente[i] = 1 - descendiente[i] # Voltear el bit (0->1, 1->0)
    # Asegurar que al menos una instancia sea seleccionada
    if np.sum(descendiente) == 0:
        idx_random = np.random.randint(0, len(descendiente))
        descendiente[idx_random] = 1
    return descendiente

# --- 5. Reemplazo ---
# Se implementará como (μ+λ): combinar padres y descendientes, y elegir los N mejores.

# --- 6. Iteración (Bucle Evolutivo) ---
print("Iniciando Proceso Evolutivo para Selección de Instancias...")

# Inicializar población
poblacion_actual = inicializar_poblacion(N_INDIVIDUOS, N_INSTANCIAS_TOTALES)
mejor_aptitud_global = -float('inf')
mejor_individuo_global = None

historial_mejores_aptitudes = []

for generacion in tqdm(range(N_GENERACIONES), desc="Generaciones"):
    # 2. Evaluación
    aptitudes = []
    for ind in poblacion_actual:
        # Para la aptitud, usamos el dataset de entrenamiento completo y su correspondiente 'y'
        # La validación se hará internamente en calcular_aptitud
        apt = calcular_aptitud(ind, X_train_full_scaled, y_train_full,
                               X_train_full_scaled, y_train_full, # Pasamos el mismo set para que la función haga un split interno
                               ALPHA_PENALIZACION)
        aptitudes.append(apt)

    # Encontrar el mejor de esta generación
    mejor_aptitud_gen = max(aptitudes)
    idx_mejor_gen = np.argmax(aptitudes)
    mejor_individuo_gen = poblacion_actual[idx_mejor_gen]
    historial_mejores_aptitudes.append(mejor_aptitud_gen)

    if mejor_aptitud_gen > mejor_aptitud_global:
        mejor_aptitud_global = mejor_aptitud_gen
        mejor_individuo_global = mejor_individuo_gen
        print(f"\nNueva mejor aptitud global en Gen {generacion}: {mejor_aptitud_global:.4f}, "
              f"Instancias: {np.sum(mejor_individuo_global)}/{N_INSTANCIAS_TOTALES}")


    # 3. Selección (para generar descendencia)
    padres_seleccionados = seleccion_por_torneo(poblacion_actual, aptitudes, TAMANO_TORNEO)

    # 4. Mutación (Generar descendientes)
    descendientes = []
    for padre in padres_seleccionados:
        descendiente = mutar(padre, PROB_MUTACION)
        descendientes.append(descendiente)

    # 5. Reemplazo (μ+λ): Combinar población actual y descendientes, y seleccionar los N mejores
    # Se evalúa la aptitud de los descendientes
    aptitudes_descendientes = []
    for desc in descendientes:
        apt_desc = calcular_aptitud(desc, X_train_full_scaled, y_train_full,
                                    X_train_full_scaled, y_train_full,
                                    ALPHA_PENALIZACION)
        aptitudes_descendientes.append(apt_desc)

    poblacion_combinada = poblacion_actual + descendientes
    aptitudes_combinadas = aptitudes + aptitudes_descendientes

    # Ordenar por aptitud (descendente) y seleccionar los N mejores
    indices_ordenados = np.argsort(aptitudes_combinadas)[::-1] # Mayor a menor
    poblacion_nueva = []
    for i in range(N_INDIVIDUOS):
        poblacion_nueva.append(poblacion_combinada[indices_ordenados[i]])
    poblacion_actual = poblacion_nueva

# --- Fin del Proceso Evolutivo ---
print("\nProceso Evolutivo Finalizado.")
print(f"Mejor aptitud encontrada: {mejor_aptitud_global:.4f}")
print(f"Número de instancias seleccionadas: {np.sum(mejor_individuo_global)} de {N_INSTANCIAS_TOTALES}")
print(f"Ratio de selección: {np.sum(mejor_individuo_global)/N_INSTANCIAS_TOTALES:.2%}")

# --- Evaluación Final del Mejor Subconjunto ---
# Entrenar un modelo con el mejor subconjunto de instancias encontrado y evaluar en el CONJUNTO DE PRUEBA
indices_finales_seleccionados = np.where(mejor_individuo_global == 1)[0]
X_train_subset_final = X_train_full_scaled[indices_finales_seleccionados]
y_train_subset_final = y_train_full[indices_finales_seleccionados]

print(f"\nEntrenando modelo final con {X_train_subset_final.shape[0]} instancias seleccionadas...")

if X_train_subset_final.shape[0] > 0:
    modelo_final = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
    modelo_final.fit(X_train_subset_final, y_train_subset_final)
    predicciones_test = modelo_final.predict(X_test_scaled)
    accuracy_test_subset = accuracy_score(y_test, predicciones_test)
    print(f"Precisión en el conjunto de prueba (con subset): {accuracy_test_subset:.4f}")
else:
    print("No se seleccionaron instancias, no se puede entrenar el modelo final.")


# Comparación: Entrenar con el dataset de entrenamiento completo
modelo_completo = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
modelo_completo.fit(X_train_full_scaled, y_train_full)
predicciones_test_completo = modelo_completo.predict(X_test_scaled)
accuracy_test_completo = accuracy_score(y_test, predicciones_test_completo)
print(f"Precisión en el conjunto de prueba (dataset completo): {accuracy_test_completo:.4f} "
      f"usando {X_train_full_scaled.shape[0]} instancias.")

# Opcional: Visualizar historial de aptitudes
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(historial_mejores_aptitudes, marker='o', linestyle='-')
plt.title('Mejor Aptitud por Generación')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.grid(True)
plt.show()