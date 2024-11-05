import numpy as np
import pandas as pd

# Parámetros del BFOA original
num_bacterias = 50       # Tamaño de población de bacterias
ciclos = 100             # Número de ciclos
paso = 0.05              # Paso de movimiento
atraccion = 0.1          # Valor de atracción
repulsion = 0.05         # Valor de repulsión

# Función de fitness
def fitness(bacteria):
    x, y = bacteria
    return np.sin(x) * np.cos(y)

# Simulación del Blosum Score
def blosum_score():
    return np.random.randint(50, 100)  # Puntaje simulado

# Ejecución del BFOA Original con Mejora Adaptativa
def ejecutar_bfoa_original(num_iteraciones):
    resultados = []
    for iteracion in range(1, num_iteraciones + 1):
        # Inicialización
        bacterias = np.random.rand(num_bacterias, 2)
        nfe = 0
        mejora_adaptativa = 0

        # Calcular el fitness inicial promedio
        fitness_inicial = np.mean([fitness(bacteria) for bacteria in bacterias])
        
        # Ejecución del algoritmo
        for ciclo in range(ciclos):
            for i, bacteria in enumerate(bacterias):
                nueva_bacteria = bacteria + paso * (np.random.rand(2) - 0.5)
                nfe += 1  # Cuenta como una evaluación de función
                if fitness(nueva_bacteria) > fitness(bacteria):
                    bacterias[i] = nueva_bacteria

                # Atracción y repulsión
                for j, otra_bacteria in enumerate(bacterias):
                    if i != j:
                        distancia = np.linalg.norm(bacteria - otra_bacteria)
                        if distancia < atraccion:
                            bacterias[i] += paso * (otra_bacteria - bacteria) * repulsion

            # Eliminación y dispersión cada 20 ciclos
            if ciclo % 20 == 0:
                bacterias = np.where(np.random.rand(num_bacterias, 2) > 0.5, bacterias, np.random.rand(num_bacterias, 2))

        # Calcular mejora adaptativa como la diferencia entre el fitness final y el fitness inicial
        fitness_final = max(bacterias, key=fitness)
        mejora_adaptativa = fitness(fitness_final) - fitness_inicial

        # Registro de datos para la iteración
        resultados.append({
            "Iteración": iteracion,
            "Mejor Fitness": fitness(fitness_final),
            "Blosum Score": blosum_score(),
            "Mejora Adaptativa": mejora_adaptativa,
            "NFE": nfe,
            "Tamaño de Población": num_bacterias
        })

    return pd.DataFrame(resultados)

# Ejecutar el algoritmo y mostrar resultados
num_iteraciones = 30
df_resultados_original = ejecutar_bfoa_original(num_iteraciones)
print(df_resultados_original)
