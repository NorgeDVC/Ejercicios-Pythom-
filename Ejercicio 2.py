"""
ejercicio para calcular métricas de rendimiento de un clasificador de spam
basado en una matriz de confusión dada
"""

class EvaluadorSpam:
    def __init__(self, verdaderos_positivos, verdaderos_negativos, falsos_positivos, falsos_negativos):
        """
        Inicializa el evaluador con los valores de la matriz de confusión
        
        Args:
            verdaderos_positivos (int): Casos de spam correctamente identificados
            verdaderos_negativos (int): Casos no spam correctamente identificados
            falsos_positivos (int): Casos no spam marcados incorrectamente como spam
            falsos_negativos (int): Casos spam marcados incorrectamente como no spam
        """
        self.VP = verdaderos_positivos
        self.VN = verdaderos_negativos
        self.FP = falsos_positivos
        self.FN = falsos_negativos
        
    def calcular_exactitud(self):
        """Calcula la exactitud (accuracy) del clasificador"""
        total = self.VP + self.VN + self.FP + self.FN
        return (self.VP + self.VN) / total if total != 0 else 0
    
    def calcular_precision(self):
        """Calcula la precisión del clasificador"""
        return self.VP / (self.VP + self.FP) if (self.VP + self.FP) != 0 else 0
    
    def calcular_sensibilidad(self):
        """Calcula la sensibilidad (recall) del clasificador"""
        return self.VP / (self.VP + self.FN) if (self.VP + self.FN) != 0 else 0
    
    def calcular_f1(self):
        """Calcula la medida F1 del clasificador"""
        precision = self.calcular_precision()
        sensibilidad = self.calcular_sensibilidad()
        return 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) != 0 else 0
    
    def generar_reporte(self):
        """Genera un reporte completo con todas las métricas"""
        print("\nREPORTE DE MÉTRICAS - CLASIFICADOR DE SPAM")
        print("----------------------------------------")
        print(f"Verdaderos Positivos (VP): {self.VP}")
        print(f"Verdaderos Negativos (VN): {self.VN}")
        print(f"Falsos Positivos (FP): {self.FP}")
        print(f"Falsos Negativos (FN): {self.FN}")
        print("\nMÉTRICAS CALCULADAS:")
        print(f"Exactitud (Accuracy): {self.calcular_exactitud
