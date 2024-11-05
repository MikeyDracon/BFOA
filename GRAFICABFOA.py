import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del BFOA
num_bacterias = 50       # Número de bacterias
ciclos = 100             # Número de ciclos por ejecución
paso = 0.05              # Paso de movimiento de las bacterias
atraccion = 0.1          # Valor inicial de atracción
repulsion = 0.05         # Valor inicial de repulsión

# Función de fitness
def fitness(bacteria):
    x, y = bacteria
    return np.sin(x) * np.cos(y)

# Algoritmo BFOA Original
def ejecutar_bfoa_original(num_iteraciones):
    resultados = []
    for iteracion in range(1, num_iteraciones + 1):
        # Inicialización de las bacterias
        bacterias = np.random.rand(num_bacterias, 2)
        
        # Ciclos del BFOA original
        for ciclo in range(ciclos):
            for i, bacteria in enumerate(bacterias):
                nueva_bacteria = bacteria + paso * (np.random.rand(2) - 0.5)
                if fitness(nueva_bacteria) > fitness(bacteria):
                    bacterias[i] = nueva_bacteria
                
                # Atracción y repulsión
                for j, otra_bacteria in enumerate(bacterias):
                    if i != j:
                        distancia = np.linalg.norm(bacteria - otra_bacteria)
                        if distancia < atraccion:
                            bacterias[i] += paso * (otra_bacteria - bacteria) * repulsion

            # Fase de eliminación y dispersión
            if ciclo % 20 == 0:
                bacterias = np.where(np.random.rand(num_bacterias, 2) > 0.5, bacterias, np.random.rand(num_bacterias, 2))
        
        # Guardar el mejor fitness de la iteración
        mejor_fitness = max(bacterias, key=fitness)
        resultados.append({"Iteración": iteracion, "Mejor Fitness": fitness(mejor_fitness)})

    return pd.DataFrame(resultados)

# Ejecutar BFOA Original
num_iteraciones = 30
resultados_original = ejecutar_bfoa_original(num_iteraciones)

# Gráfico de líneas para el BFOA Original
plt.figure(figsize=(10, 6))
plt.plot(resultados_original["Iteración"], resultados_original["Mejor Fitness"], label="BFOA Original", marker="o", color="blue")
plt.title("Mejor Fitness por Iteración para BFOA Original")
plt.xlabel("Iteración")
plt.ylabel("Mejor Fitness")
plt.legend()
plt.grid(True)
plt.show()
