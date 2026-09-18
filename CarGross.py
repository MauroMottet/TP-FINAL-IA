"""
CarGross.py - Implementación del Clasificador No Supervisado de Carpenter/Grossberg (ART-1)
Referencia Teórica: Richard P. Lippmann (1987), 'An Introduction to Computing with Neural Nets', Box 3.
Trabajo Final Integrador - Cátedra de Redes Neuronales / IDTI Lab - UADER
"""

import sys
import os
import argparse
import csv
from typing import List, Tuple

def parse_arguments():
    """Configura el parser de argumentos de línea de comandos y el manual de usuario (man)."""
    parser = argparse.ArgumentParser(
        prog="CarGross.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
==============================================================================
   RED DE CARPENTER/GROSSBERG (ART-1) - CLASIFICADOR NO SUPERVISADO
==============================================================================
Este script implementa el algoritmo secuencial del líder basado en la red de
Carpenter/Grossberg para clustering no supervisado sobre patrones estrictamente binarios.

REGLAS DE OPERACIÓN:
  1. El archivo de entrada debe ser un CSV con valores 0 o 1.
  2. El parámetro de vigilancia (rho) debe estar en el intervalo [0.0, 1.0].
  3. No se admiten vectores completamente nulos (suma = 0) para evitar
     indeterminación matemática en el test de vigilancia (división por cero).
        """,
        epilog="""
EJEMPLOS DE USO:
  python CarGross.py -d dataset_telemetria_base.csv -p 0.6
  python CarGross.py --dataset dataset_telemetria_estres.csv --rho 0.8 --verbose
        """
    )
    
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Ruta al archivo CSV de entrada con variables binarias.")
    parser.add_argument("-p", "--rho", type=float, default=0.5,
                        help="Umbral de vigilancia rho en [0.0, 1.0]. Por defecto: 0.5.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Muestra la asignación paso a paso de cada vector.")
    
    return parser

def load_and_validate_csv(file_path: str) -> Tuple[List[str], List[List[int]]]:
    """Carga y valida rigurosamente el dataset CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: El archivo '{file_path}' no existe.")
        
    data = []
    headers = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("Error: El archivo CSV está completamente vacío.")
            
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            int_row = []
            for col_idx, val in enumerate(row):
                val_clean = val.strip()
                if val_clean not in ('0', '1'):
                    raise ValueError(
                        f"Valor no binario detectado en fila {row_idx}, col {col_idx+1}: '{val}'. "
                        f"Carpenter/Grossberg solo admite entradas 0 o 1."
                    )
                int_row.append(int(val_clean))
            
            if sum(int_row) == 0:
                raise ValueError(
                    f"Fila {row_idx} contiene un vector nulo (todos 0). "
                    f"ART-1 requiere ||X|| > 0 para calcular ||T*X|| / ||X||."
                )
            data.append(int_row)
            
    if len(data) == 0:
        raise ValueError("Error: No se encontraron filas de datos válidas en el CSV.")
    return headers, data

def run_carpenter_grossberg(data: List[List[int]], rho: float, verbose: bool = False):
    """
    Ejecución del algoritmo según Lippmann (1987, Box 3):
      N: número de variables de entrada
      b: matriz de pesos ascendentes (bottom-up)
      t: matriz de pesos descendentes (top-down / prototipos)
    """
    N = len(data[0])
    b: List[List[float]] = []
    t: List[List[float]] = []
    cluster_assignments: List[int] = []
    
    for idx, x in enumerate(data):
        norm_x = float(sum(x))
        
        # Si aún no hay clústeres creados, el primer patrón inicializa el primer nodo
        if len(b) == 0:
            t.append([float(val) for val in x])
            b.append([float(val) / (0.5 + norm_x) for val in x])
            cluster_assignments.append(0)
            if verbose:
                print(f"Patrón {idx+1:03d} -> Inicializa Clúster 0")
            continue
            
        M = len(b)
        # Paso 3: Calcular matching scores mu_j = sum(b_ij * x_i)
        scores = [sum(b[j][i] * x[i] for i in range(N)) for j in range(M)]
        disabled = set()
        matched = False
        
        # Búsqueda secuencial del mejor ejemplar con inhibición lateral
        while len(disabled) < M:
            # Paso 4: Selección del ganador
            available_nodes = [j for j in range(M) if j not in disabled]
            best_j = max(available_nodes, key=lambda j: scores[j])
            
            # Paso 5: Test de vigilancia: ||T_j * X|| / ||X|| > rho
            dot_product = sum(t[best_j][i] * x[i] for i in range(N))
            vigilance_ratio = dot_product / norm_x
            
            if vigilance_ratio > rho:
                # Paso 7: Adaptar ejemplar (operación lógica AND)
                t[best_j] = [t[best_j][i] * x[i] for i in range(N)]
                sum_t = sum(t[best_j])
                b[best_j] = [t[best_j][i] / (0.5 + sum_t) for i in range(N)]
                cluster_assignments.append(best_j)
                matched = True
                if verbose:
                    print(f"Patrón {idx+1:03d} -> Asignado a Clúster {best_j} (Vigilancia: {vigilance_ratio:.3f})")
                break
            else:
                # Paso 6: Inhabilitar ejemplar transitoriamente
                disabled.add(best_j)
                
        if not matched:
            # Creación de nueva categoría (nuevo nodo de salida)
            new_cluster_id = M
            t.append([float(val) for val in x])
            b.append([float(val) / (0.5 + norm_x) for val in x])
            cluster_assignments.append(new_cluster_id)
            if verbose:
                print(f"Patrón {idx+1:03d} -> Crea NUEVO Clúster {new_cluster_id} (Rechazado por vigilancia)")
                
    return cluster_assignments, t

def main():
    parser = parse_arguments()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not (0.0 <= args.rho <= 1.0):
        print(f"\n[ERROR] Parámetro de vigilancia fuera de rango: rho={args.rho}. Debe ser entre 0.0 y 1.0.\n")
        parser.print_help()
        sys.exit(1)
        
    try:
        headers, data = load_and_validate_csv(args.dataset)
    except Exception as e:
        print(f"\n[ERROR EN LA CARGA]: {e}")
        sys.exit(1)
        
    print(f"\n--- Ejecutando Carpenter/Grossberg en '{args.dataset}' ---")
    print(f"Instancias: {len(data)} | Variables (N): {len(headers)} | Umbral de Vigilancia (rho): {args.rho}\n")
    
    assignments, exemplars = run_carpenter_grossberg(data, args.rho, args.verbose)
    
    # Resumen de resultados
    unique_clusters = sorted(list(set(assignments)))
    print("\n================ RESUMEN DE AGRUPAMIENTO ================")
    print(f"Total de clústeres formados (M): {len(unique_clusters)}")
    print("-" * 57)
    for c in unique_clusters:
        count = assignments.count(c)
        pct = (count / len(data)) * 100
        proto = [int(v) for v in exemplars[c]]
        print(f"Clúster {c:02d}: {count:3d} instancias ({pct:5.1f}%) | Prototipo Top-Down: {proto}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()