# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:53:46 2025

@author: MARCELOFGB
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
matplotlib.use('Qt5Agg')
# --- Parámetros de la simulación ---
AREA_SIZE = 10
NUM_SENTINELS = 15
NUM_AERODESI = 3
MAX_STEPS = 100
VISUAL_RANGE = 3.0
SPEED = 0.8
NEUTRALIZATION_THRESHOLD = 10

SEARCH = 0
SWARM = 1

FOLLOW = 2
RANDOM = 3

ACTIVE = 0
NEUTRALIZED = 1

class Sentinel:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.behavior = SEARCH
        self.target_x = None
        self.target_y = None

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x = max(0, min(AREA_SIZE - 1, self.x))
        self.y = max(0, min(AREA_SIZE - 1, self.y))

    def get_position(self):
        return (self.x, self.y)

class Aerodeslizador:
    def __init__(self, id, x, y, is_decoy=False):
        self.id = id
        self.x = x
        self.y = y
        self.status = ACTIVE
        self.is_decoy = is_decoy
        self.nearby_sentinels = 0

    def get_position(self):
        return (self.x, self.y)

    def set_status(self, status):
        self.status = status

    def update_nearby_sentinels(self, sentinels):
        self.nearby_sentinels = 0
        for sentinel in sentinels:
            dist = np.linalg.norm(np.array(sentinel.get_position()) - np.array([self.x, self.y]))
            if dist < VISUAL_RANGE:
                self.nearby_sentinels += 1

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def calculate_centroid(sentinels):
    if not sentinels:
        return (0, 0)
    sum_x = sum(s.x for s in sentinels)
    sum_y = sum(s.y for s in sentinels)
    return (sum_x / len(sentinels), sum_y / len(sentinels))

def update_sentinel_behavior(sentinel, active_aerodeslizadores, all_sentinels):
    closest_aerodeslizador = None
    min_dist_to_aerodeslizador = float('inf')

    for aerodeslizador in active_aerodeslizadores:
        dist = calculate_distance(sentinel.get_position(), aerodeslizador.get_position())
        if dist < min_dist_to_aerodeslizador:
            min_dist_to_aerodeslizador = dist
            closest_aerodeslizador = aerodeslizador

    DETECTION_THRESHOLD = VISUAL_RANGE * 1.5
    if closest_aerodeslizador and min_dist_to_aerodeslizador < DETECTION_THRESHOLD:
        sentinel.behavior = SWARM
        sentinel.target_x, sentinel.target_y = closest_aerodeslizador.get_position()
    elif sentinel.behavior == SWARM and sentinel.target_x is not None:
        pass 
    else:
        sentinel.behavior = SEARCH

    if sentinel.behavior == SWARM and sentinel.target_x is not None:
        target_pos = np.array([sentinel.target_x, sentinel.target_y])
        current_pos = np.array([sentinel.x, sentinel.y])

        sentinels_for_target = [s for s in all_sentinels if s.behavior == SWARM and s.target_x == sentinel.target_x and s.target_y == sentinel.target_y]
        
        centroid_pos = np.array([0.0,0.0])
        if sentinels_for_target:
            centroid_x, centroid_y = calculate_centroid(sentinels_for_target)
            centroid_pos = np.array([centroid_x, centroid_y])
        else:
            centroid_pos = target_pos 

        move_vector = np.array([0.0, 0.0])
        
        direction_to_centroid = centroid_pos - current_pos
        if np.linalg.norm(direction_to_centroid) > 0.1: 
            direction_to_centroid /= np.linalg.norm(direction_to_centroid)
            move_vector += direction_to_centroid * SPEED * 0.7

        direction_to_target = target_pos - current_pos
        if np.linalg.norm(direction_to_target) > 0.1: 
            direction_to_target /= np.linalg.norm(direction_to_target)
            move_vector += direction_to_target * SPEED * 0.3

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector) * SPEED 

        sentinel.move(move_vector[0], move_vector[1])

    elif sentinel.behavior == SEARCH:
        angle = np.random.uniform(0, 2 * np.pi)
        move_dist = SPEED * np.random.uniform(0.5, 1.2)
        dx = move_dist * np.cos(angle)
        dy = move_dist * np.sin(angle)
        sentinel.move(dx, dy)

# --- Inicialización ---
sentinels = []
for i in range(NUM_SENTINELS):
    sentinels.append(Sentinel(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE)))

aerodeslizadores = []
decoy_index = np.random.randint(0, NUM_AERODESI)
for i in range(NUM_AERODESI):
    aerodeslizadores.append(Aerodeslizador(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), is_decoy=(i == decoy_index)))

print("Simulación iniciada. Centinelas: {}, Aerodeslizadores: {}".format(NUM_SENTINELS, NUM_AERODESI))

# --- Configuración para Spyder ---
# No necesitamos switch_backend si ya estamos en Spyder, pero podemos activar ion()
plt.ion() 

# Crear la figura y ejes una vez al inicio
fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Simulación Bioinspirada - Spyder")
ax.grid(True)

# Inicializar los objetos gráficos que vamos a actualizar
# Centinelas
sentinel_plot, = ax.plot([], [], 'bo', markersize=3, label='Centinelas')

# Aerodeslizadores (guardamos los objetos de plot y texto por separado)
aerodeslizador_plots_data = [] # Contendrá diccionarios con plot_obj, text_obj, label, etc.
legend_handles = []

for i, aero in enumerate(aerodeslizadores):
    marker = 'D' if aero.is_decoy else '^'
    color = 'gray' if aero.is_decoy else 'red'
    label = 'Aerodeslizador Señuelo' if aero.is_decoy else 'Aerodeslizador Real'
    
    plot_obj, = ax.plot([], [], marker=marker, color=color, markersize=10, label=label)
    text_obj = ax.text(0, 0, "", color='black', fontsize=8, ha='center') # Texto inicial vacío
    
    aerodeslizador_plots_data.append({
        'plot_obj': plot_obj, 
        'text_obj': text_obj,
        'label': label, 
        'is_decoy': aero.is_decoy, 
        'id': aero.id,
        'color': color,
        'marker': marker
    })
    
    # Añadir elementos de leyenda únicos
    if label not in [h.get_label() for h in legend_handles]:
        legend_handles.append(plt.Line2D([], [], marker=marker, color=color, linestyle='None', markersize=10, label=label))

# Crear la leyenda una vez
legend = ax.legend(handles=legend_handles, loc='upper right')

def update_graphics(step, sentinels, aerodeslizadores, sentinel_plot, aerodeslizador_plots_data, ax, legend):
    # --- Actualizar Centinelas ---
    sentinel_plot.set_data([s.x for s in sentinels], [s.y for s in sentinels])

    # --- Actualizar Aerodeslizadores ---
    for i, aero in enumerate(aerodeslizadores):
        data = aerodeslizador_plots_data[i]
        plot_obj = data['plot_obj']
        text_obj = data['text_obj']
        
        if aero.status == ACTIVE:
            plot_obj.set_visible(True)
            plot_obj.set_xdata([aero.x])
            plot_obj.set_ydata([aero.y])
            
            text_obj.set_text(str(aero.nearby_sentinels))
            text_obj.set_position((aero.x, aero.y + 0.2))
            text_obj.set_visible(True)
        else:
            plot_obj.set_visible(False)
            text_obj.set_visible(False)

    ax.set_title(f"Paso de Simulación: {step}")
    
    # Dibujar la escena y actualizar la ventana de Spyder
    fig.canvas.draw_idle() # Dibuja si es necesario
    fig.canvas.flush_events() # Procesa eventos de la GUI (crucial para Spyder)

# --- Bucle de Simulación ---
for step in range(MAX_STEPS):
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]

    if not active_aerodeslizadores:
        print("Todos los aerodeslizadores han sido neutralizados.")
        break

    for aero in active_aerodeslizadores:
        aero.update_nearby_sentinels(sentinels)
        if aero.nearby_sentinels >= NEUTRALIZATION_THRESHOLD:
            if not aero.is_decoy:
                aero.set_status(NEUTRALIZED)
                print(f"Aerodeslizador {aero.id} (Real) neutralizado por {aero.nearby_sentinels} centinelas.")
            else:
                print(f"Aerodeslizador Señuelo {aero.id} neutralizado por {aero.nearby_sentinels} centinelas. Buscando otro.")
                aero.set_status(NEUTRALIZED) 
                
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]
    
    if not active_aerodeslizadores:
        print("Todos los aerodeslizadores (incluyendo señuelos) han sido 'neutralizados'.")
        update_graphics(step, sentinels, aerodeslizadores, sentinel_plot, aerodeslizador_plots_data, ax, legend) 
        break

    for sentinel in sentinels:
        update_sentinel_behavior(sentinel, active_aerodeslizadores, sentinels)

    # Actualizar los gráficos en la ventana de Spyder
    update_graphics(step, sentinels, aerodeslizadores, sentinel_plot, aerodeslizador_plots_data, ax, legend)
    
    # Pausa breve para permitir la visualización y procesamiento de eventos de Spyder
    time.sleep(0.8) 

else: 
    print("Se alcanzó el número máximo de pasos de simulación.")

print("\nSimulación finalizada.")

# Al finalizar, mantén la última gráfica visible en Spyder
plt.show()