import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del BFOA mejorado
num_bacterias = 50       # Tamaño de población de bacterias
ciclos = 100             # Número de ciclos
paso = 0.05              # Paso de movimiento
atraccion_inicial = 0.1  # Valor inicial de atracción
repulsion_inicial = 0.05 # Valor inicial de repulsión

# Función de fitness
def fitness(bacteria):
    x, y = bacteria
    return np.sin(x) * np.cos(y)

# Simulación del Blosum Score
def blosum_score():
    return np.random.randint(50, 100)  # Puntaje simulado

# Ejecución del BFOA Mejorado con Mejora Adaptativa
def ejecutar_bfoa_mejorado(num_iteraciones):
    resultados = []
    for iteracion in range(1, num_iteraciones + 1):
        # Inicialización de bacterias y parámetros adaptativos
        bacterias = np.random.rand(num_bacterias, 2)
        atraccion = atraccion_inicial
        repulsion = repulsion_inicial
        nfe = 0
        mejora_adaptativa = 0

        # Calcular el fitness inicial promedio
        fitness_inicial = np.mean([fitness(bacteria) for bacteria in bacterias])
        
        # Ejecución del algoritmo en cada ciclo
        for ciclo in range(ciclos):
            # Ajuste adaptativo cada 10 ciclos
            if ciclo % 10 == 0 and ciclo != 0:
                atraccion *= 1.05
                repulsion *= 0.95

            for i, bacteria in enumerate(bacterias):
                nueva_bacteria = bacteria + paso * (np.random.rand(2) - 0.5)
                nfe += 1  # Cuenta como una evaluación de función
                if fitness(nueva_bacteria) > fitness(bacteria):
                    bacterias[i] = nueva_bacteria

                # Interacción entre bacterias con atracción y repulsión adaptativos
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

# Ejecutar el algoritmo mejorado y mostrar los resultados
num_iteraciones = 30
df_resultados_mejorado = ejecutar_bfoa_mejorado(num_iteraciones)
print(df_resultados_mejorado)

# Gráfico de líneas para el BFOA Mejorado
plt.figure(figsize=(10, 6))
plt.plot(df_resultados_mejorado["Iteración"], df_resultados_mejorado["Mejor Fitness"], label="BFOA Mejorado", marker="o", color="green")
plt.title("Mejor Fitness por Iteración para BFOA Mejorado")
plt.xlabel("Iteración")
plt.ylabel("Mejor Fitness")
plt.legend()
plt.grid(True)
plt.show()
