import numpy as np

class ComparadorVectores:
    def __init__(self):
        # Diccionario que mapea nombres de distancia a métodos de la clase
        self.distancias = {
            'euclidea': self.distancia_euclidea,
            'euclidea_normalizada': self.distancia_euclidea_normalizada,
            'chebychev': self.distancia_chebychev,
            'manhattan': self.distancia_manhattan
        }
    
    def encontrar_similares(self, funcion_distancia, nuevo_vector, base_datos, k=None):
        """
        Encuentra los vectores más similares en la base de datos según la función de distancia especificada.
        
        Args:
            funcion_distancia (str): Nombre de la función de distancia a usar
            nuevo_vector (list or np.array): Vector de características a comparar
            base_datos (list of lists or np.array): Matriz de vectores de características
            k (int, optional): Número de resultados más similares a devolver. Si es None, devuelve todos ordenados.
            
        Returns:
            list: Lista de vectores más similares ordenados por distancia ascendente
        """
        # Verificar que la función de distancia existe
        if funcion_distancia not in self.distancias:
            raise ValueError(f"Función de distancia no soportada. Opciones: {list(self.distancias.keys())}")
        
        # Obtener la función de distancia usando reflexión
        funcion_dist = self.distancias[funcion_distancia]
        
        # Calcular distancias entre el nuevo vector y todos los de la base de datos
        distancias = []
        for i, vector in enumerate(base_datos):
            dist = funcion_dist(nuevo_vector, vector)
            distancias.append((dist, i))
        
        # Ordenar por distancia ascendente
        distancias.sort(key=lambda x: x[0])
        
        # Obtener los índices de los k más cercanos (o todos si k es None)
        if k is not None:
            indices = [idx for dist, idx in distancias[:k]]
        else:
            indices = [idx for dist, idx in distancias]
        
        # Devolver los vectores más similares
        return [base_datos[i] for i in indices]
    
    def distancia_euclidea(self, a, b):
        """Distancia Euclidea estándar"""
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum((a - b)**2))
    
    def distancia_euclidea_normalizada(self, a, b):
        """Distancia Euclidea normalizada por la longitud de los vectores"""
        a = np.array(a)
        b = np.array(b)
        diff = a - b
        norm_diff = diff / np.linalg.norm(diff, ord=2)
        return np.linalg.norm(norm_diff, ord=2)
    
    def distancia_chebychev(self, a, b):
        """Distancia de Chebychev (máxima diferencia absoluta entre componentes)"""
        a = np.array(a)
        b = np.array(b)
        return np.max(np.abs(a - b))
    
    def distancia_manhattan(self, a, b):
        """Distancia de Manhattan (suma de diferencias absolutas)"""
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.abs(a - b))


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del comparador
    comparador = ComparadorVectores()
    
    # Base de datos de ejemplo (similar a pacientes con características médicas)
    base_datos = [
        [65, 180, 120, 80],  # paciente 1
        [70, 170, 130, 85],  # paciente 2
        [60, 175, 110, 75],   # paciente 3
        [68, 182, 125, 78],   # paciente 4
        [72, 168, 135, 90]    # paciente 5
    ]
    
    # Nuevo vector a comparar (nuevo paciente)
    nuevo_paciente = [67, 175, 125, 82]
    
    # Probar diferentes distancias
    distancias = ['euclidea', 'euclidea_normalizada', 'chebychev', 'manhattan']
    
    for distancia in distancias:
        print(f"\nVectores más similares usando distancia {distancia}:")
        similares = comparador.encontrar_similares(distancia, nuevo_paciente, base_datos, k=3)
        for i, vector in enumerate(similares):
            print(f"{i+1}. {vector}")
