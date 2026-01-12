#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Nombre del archivo de entrada
FILENAME = 'trajectory.csv'

def main():
    # 1. Verificar si el archivo existe
    if not os.path.exists(FILENAME):
        print(f"ERROR: No se encuentra el archivo '{FILENAME}'.")
        print("Asegúrate de haber movido el archivo con: mv ~/.ros/trajectory.csv .")
        return

    print(f"Cargando datos de {FILENAME}...")
    df = pd.read_csv(FILENAME)

    # 2. Procesamiento de datos
    # Ajustar tiempo para que empiece en 0
    df['time_rel'] = df['time'] - df['time'].iloc[0]

    # Calcular diferencias (delta) entre cada punto
    df['dt'] = df['time_rel'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()

    # Distancia recorrida en cada paso (Hipotenusa)
    df['dist_step'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # --- Cálculo de Velocidad (v = d / t) ---
    # Manejamos la división por cero poniendo NaN si dt es 0
    df['velocity'] = df['dist_step'] / df['dt']

    # Suavizado: Los datos crudos (especialmente derivadas) tienen ruido.
    # Usamos una media móvil (rolling mean) de 10 muestras para limpiar la gráfica.
    df['velocity_smooth'] = df['velocity'].rolling(window=10, center=True).mean()

    # --- Cálculo de Aceleración (a = dv / t) ---
    # Usamos la velocidad suavizada para calcular la aceleración
    df['dv'] = df['velocity_smooth'].diff()
    df['acceleration'] = df['dv'] / df['dt']

    # 3. Configuración de las Gráficas
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle('Análisis de Movimiento del Turtlebot', fontsize=16)

    # --- Gráfica 1: Trayectoria (X vs Y) ---
    # Ocupa la mitad superior
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['x'], df['y'], label='Recorrido', color='#1f77b4', linewidth=2)
    # Marcar inicio y fin
    ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', label='Inicio', s=100, zorder=5)
    ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', label='Fin', s=100, zorder=5)
    
    ax1.set_title("Trayectoria en el plano (Odom)")
    ax1.set_xlabel("Posición X [m]")
    ax1.set_ylabel("Posición Y [m]")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axis('equal') # Importante para no deformar la realidad

    # --- Gráfica 2: Velocidad Lineal ---
    # Ocupa parte inferior izquierda
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(df['time_rel'], df['velocity_smooth'], color='#ff7f0e', label='Velocidad (m/s)')
    ax2.set_title("Velocidad Lineal")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Velocidad [m/s]")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Gráfica 3: Aceleración Lineal ---
    # Ocupa parte inferior derecha
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(df['time_rel'], df['acceleration'], color='#2ca02c', label='Aceleración (m/s²)')
    ax3.set_title("Aceleración Lineal")
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Aceleración [m/s²]")
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. Guardar y Mostrar
    plt.tight_layout()
    
    # Guardamos la imagen por si falla la ventana gráfica
    output_img = "resultados_robot.png"
    plt.savefig(output_img)
    print(f"Gráfica guardada como imagen en: {output_img}")

    # Intentamos mostrar la ventana
    try:
        print("Intentando abrir ventana gráfica...")
        plt.show()
    except Exception as e:
        print("No se pudo abrir la ventana gráfica (habitual en Docker).")
        print(f"Por favor, abre el archivo '{output_img}' manualmente.")

if __name__ == "__main__":
    main()