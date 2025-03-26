# Importaciones necesarias
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Definición de los datos de entrenamiento
datos_entrenamiento = {
    'X1': [2, 4, 1, 2, 2, 2, 3, 3],
    'X2': [0, 4, 1, 4, 2, 3, 4, 3],
    'Etiqueta': [0, 1, 0, 1, 0, 1, 0, 1]
}

# Convertir a DataFrame
df_entrenamiento = pd.DataFrame(datos_entrenamiento)

# Separar características y etiquetas
caracteristicas = df_entrenamiento[['X1', 'X2']].values
clases = df_entrenamiento['Etiqueta'].values

# Configurar el modelo de vecinos más cercanos
modelo_vecinos = NearestNeighbors(n_neighbors=1, metric='manhattan')
modelo_vecinos.fit(caracteristicas)

# Punto a evaluar
nuevo_ejemplo = [[2.5, 2.5]]

# Encontrar el vecino más cercano
distancias, indices = modelo_vecinos.kneighbors(nuevo_ejemplo)

# Obtener la clase del vecino más cercano
clase_resultado = clases[indices[0][0]]

# Mostrar el resultado
print(f"Para el punto {nuevo_ejemplo[0]}, la clase predicha es: {clase_resultado}")
print(f"El vecino más cercano está en la posición {indices[0][0]+1} del conjunto de entrenamiento")
print(f"Con una distancia Manhattan de: {distancias[0][0]:.2f}")
