"""
CarGross.py - Implementación del Clasificador No Supervisado de Carpenter/Grossberg (ART-1)
Referencia Teórica: Richard P. Lippmann (1987), 'An Introduction to Computing with Neural Nets', Box 3.
Trabajo Final Integrador "Museo de Redes Neuronales" - Inteligencia Artificial
Licenciatura en Sistemas de Información - FCyT UADER / IDTI Lab

Uso rápido:
    python3 CarGross.py -d dataset_telemetria_base.csv -p 0.5
    python3 CarGross.py --help

Solo requiere la biblioteca estándar de Python (>= 3.8).
"""

import sys
import os
import argparse
import csv
from typing import List, Optional, Tuple

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
  2. La primera fila del CSV es el encabezado (nombres de las variables) y
     todas las filas deben tener la misma cantidad de columnas, separadas por coma.
  3. El parámetro de vigilancia (rho) debe estar en el intervalo [0.0, 1.0).
     El test de Lippmann es estricto (||T*X||/||X|| > rho): con rho = 1 ningún
     patrón superaría el test y cada fila quedaría sin clasificar.
  4. No se admiten vectores completamente nulos (suma = 0) para evitar
     indeterminación matemática en el test de vigilancia (división por cero).
  5. La red dispone de M nodos de salida (F2). Si se agotan, los patrones que no
     encajan en ninguna categoría se informan como "sin clasificar".
        """,
        epilog="""
EJEMPLOS DE USO:
  python3 CarGross.py -d dataset_telemetria_base.csv -p 0.6
  python3 CarGross.py --dataset dataset_telemetria_estres.csv --rho 0.8 --verbose
  python3 CarGross.py -d dataset_telemetria_base.csv -p 0.5 -o asignaciones.csv

SALIDA CSV (-o): columnas 'fila,cluster'. 'fila' es el número de registro de
datos (1 = primera fila después del encabezado) y 'cluster' es el nodo F2
asignado (-1 = sin clasificar por falta de nodos libres).
        """
    )

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Ruta al archivo CSV de entrada con variables binarias.")
    parser.add_argument("-p", "--rho", type=float, default=0.5,
                        help="Umbral de vigilancia rho en [0.0, 1.0). Por defecto: 0.5.")
    parser.add_argument("-m", "--max-nodos", type=int, default=None,
                        help="Cantidad M de nodos de salida F2. Por defecto: uno por "
                             "patrón (la red nunca se queda sin nodos).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Ruta de un CSV donde guardar la asignación fila -> clúster.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Muestra la asignación paso a paso de cada vector.")
    parser.add_argument("--fuzzy", action="store_true",
                        help="(Anexo II) Ejecuta Fuzzy C-Means (scikit-fuzzy) con el mismo "
                             "número de clusters que estabilizó ART-1 y presenta una tabla "
                             "comparativa. Requiere: pip install scikit-fuzzy numpy")
    parser.add_argument("--fuzzy-c", type=int, default=None, dest="fuzzy_c",
                        help="Fuerza el número de clusters C para Fuzzy C-Means "
                             "(por defecto: igual al número de categorías que creó ART-1). "
                             "Activa --fuzzy implícitamente.")

    # Los errores de argumentos (valor inválido, falta -d, etc.) también muestran el man
    parser.error = lambda message: fail(parser, message)
    return parser

def load_and_validate_csv(file_path: str) -> Tuple[List[str], List[List[int]]]:
    """Carga y valida rigurosamente el dataset CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        
    if os.path.isdir(file_path):
        raise ValueError(f"'{file_path}' es un directorio, no un archivo CSV.")

    data = []
    headers = []
    # utf-8-sig descarta el BOM que agrega Excel al guardar como "CSV UTF-8"
    with open(file_path, mode='r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        try:
            headers = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError("El archivo CSV está completamente vacío.")

        if len(headers) == 1 and ';' in headers[0]:
            raise ValueError(
                "El archivo parece usar ';' como separador (CSV regional de Excel). "
                "Guárdelo separado por comas (',')."
            )
        if any(h == '' for h in headers):
            raise ValueError("El encabezado contiene columnas sin nombre.")

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(v.strip() == '' for v in row):
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"Fila {row_idx} tiene {len(row)} columnas, pero el encabezado "
                    f"define {len(headers)} variables."
                )
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
        raise ValueError("No se encontraron filas de datos válidas en el CSV.")
    return headers, data

def run_carpenter_grossberg(data: List[List[int]], rho: float, verbose: bool = False,
                            max_nodes: Optional[int] = None) -> Tuple[List[int], List[List[float]]]:
    """
    Ejecución del algoritmo según Lippmann (1987, Box 3):
      N: número de variables de entrada (nodos de la capa F1)
      M: número de nodos de salida disponibles (capa F2)
      b: matriz de pesos ascendentes (bottom-up)
      t: matriz de pesos descendentes (top-down / prototipos)

    Paso 1 del Box 3: todos los nodos arrancan "libres" con t = 1 y
    b = 1/(1+N), y compiten desde el principio en la selección del ganador.
    Como todos los nodos libres son idénticos, alcanza con representarlos
    con un único candidato libre (el de menor índice, que es el que ganaría
    la maximización); el resultado es el mismo y el costo no crece con M.
    Un nodo libre que gana y supera la vigilancia pasa a ser una categoría
    nueva. Si todos los nodos quedan inhabilitados (Paso 6), el patrón se
    informa como sin clasificar (-1).

    Devuelve la asignación de cada patrón y los prototipos top-down t de
    los nodos comprometidos (categorías creadas), en orden de creación.
    """
    N = len(data[0])
    M = max_nodes if max_nodes is not None else len(data)
    
    # Paso 1: Inicialización de un nodo libre
    t_free = [1.0] * N
    b_free = [1.0 / (1 + N)] * N
    
    # Pesos de los nodos comprometidos; el nodo j de estas listas es el nodo F2 j
    b: List[List[float]] = []
    t: List[List[float]] = []
    cluster_assignments: List[int] = []
    
    # Paso 2: Aplicar cada nueva entrada
    for idx, x in enumerate(data):
        norm_x = float(sum(x))
        
        # Candidatos: nodos comprometidos y, si quedan, el primer nodo libre (índice len(t))
        n_candidates = len(t) + (1 if len(t) < M else 0)
        
        def weights(j: int) -> Tuple[List[float], List[float]]:
            return (b[j], t[j]) if j < len(t) else (b_free, t_free)
        
        # Paso 3: Calcular matching scores mu_j = sum(b_ij * x_i)
        scores = [sum(weights(j)[0][i] * x[i] for i in range(N)) for j in range(n_candidates)]
        disabled = set()
        assigned = -1
        
        # Búsqueda del mejor ejemplar con inhibición lateral (Pasos 4 a 6)
        while len(disabled) < n_candidates:
            # Paso 4: Selección del ganador (ante empate gana el menor índice)
            available_nodes = [j for j in range(n_candidates) if j not in disabled]
            best_j = max(available_nodes, key=lambda j: scores[j])
            t_j = weights(best_j)[1]
            
            # Paso 5: Test de vigilancia: ||T_j * X|| / ||X|| > rho
            dot_product = sum(t_j[i] * x[i] for i in range(N))
            vigilance_ratio = dot_product / norm_x
            
            if vigilance_ratio > rho:
                # Paso 7: Adaptar ejemplar (operación lógica AND)
                new_t = [t_j[i] * x[i] for i in range(N)]
                sum_t = sum(new_t)
                new_b = [new_t[i] / (0.5 + sum_t) for i in range(N)]
                if best_j < len(t):
                    t[best_j], b[best_j] = new_t, new_b
                    if verbose:
                        print(f"Patrón {idx+1:03d} -> Asignado a Clúster {best_j} (Vigilancia: {vigilance_ratio:.3f})")
                else:
                    # El nodo libre ganador se compromete como nueva categoría
                    t.append(new_t)
                    b.append(new_b)
                    if verbose:
                        print(f"Patrón {idx+1:03d} -> Crea NUEVO Clúster {best_j}")
                assigned = best_j
                break
            else:
                # Paso 6: Inhabilitar ejemplar transitoriamente
                disabled.add(best_j)
                
        if assigned == -1 and verbose:
            print(f"Patrón {idx+1:03d} -> SIN CLASIFICAR (no quedan nodos F2 libres)")
        cluster_assignments.append(assigned)
        # Paso 8: los nodos inhabilitados se rehabilitan para el próximo patrón
                
    return cluster_assignments, t

def write_assignments_csv(file_path: str, assignments: List[int]) -> None:
    """Guarda la asignación de cada patrón (1 = primera fila de datos) a su nodo F2."""
    with open(file_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["fila", "cluster"])
        for idx, c in enumerate(assignments, start=1):
            writer.writerow([idx, c])

def run_fuzzy_cmeans(
    data: List[List[int]], c: int
) -> Tuple[List[int], List[List[float]], List[List[float]]]:
    """
    Ejecuta Fuzzy C-Means (FCM) como rutina comparativa con ART-1 (Anexo II).
    Referencia: Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective
    Function Algorithms. Plenum Press.

    Parámetros
    ----------
    data : lista de vectores binarios (N_patrones × N_variables)
    c    : número de clusters C (≥ 2)

    Retorna
    -------
    (hard_assignments, centers, u_matrix)
      hard_assignments : índice del cluster dominante para cada patrón
      centers          : centros continuos — lista de C listas de N floats
      u_matrix         : u_matrix[k][i] = pertenencia del patrón i al cluster k

    Requiere
    --------
    pip install scikit-fuzzy numpy
    """
    try:
        import skfuzzy as fuzz
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "La opción --fuzzy requiere scikit-fuzzy y numpy.\n"
            "Instálelos con:\n"
            "  pip install scikit-fuzzy numpy\n"
            f"(error original: {exc})"
        ) from exc

    data_np = np.array(data, dtype=float).T   # FCM espera shape (features, samples)
    cntr, u, _, _, _, _, _ = fuzz.cmeans(
        data_np, c=c, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
    )
    hard_assignments: List[int] = [int(k) for k in np.argmax(u, axis=0)]
    centers: List[List[float]] = cntr.tolist()
    u_matrix: List[List[float]] = u.tolist()   # u_matrix[k][i]
    return hard_assignments, centers, u_matrix


def fail(parser: argparse.ArgumentParser, message: str) -> None:
    """Corrida infructuosa: informa el error, muestra el man y termina con código 1."""
    print(f"\n[ERROR] {message}\n")
    parser.print_help()
    sys.exit(1)

def main():
    parser = parse_arguments()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not (0.0 <= args.rho < 1.0):
        fail(parser, f"Parámetro de vigilancia fuera de rango: rho={args.rho}. Debe cumplir 0.0 <= rho < 1.0.")
    if args.max_nodos is not None and args.max_nodos < 1:
        fail(parser, f"La cantidad de nodos F2 debe ser al menos 1: max-nodos={args.max_nodos}.")
        
    try:
        headers, data = load_and_validate_csv(args.dataset)
    except UnicodeDecodeError:
        fail(parser, f"No se pudo leer '{args.dataset}': el archivo no es texto UTF-8 (¿es realmente un CSV?).")
    except (OSError, ValueError, csv.Error) as e:
        fail(parser, f"Carga del dataset: {e}")
        
    assignments, exemplars = run_carpenter_grossberg(data, args.rho, args.verbose, args.max_nodos)
    
    if args.output:
        try:
            write_assignments_csv(args.output, assignments)
        except OSError as e:
            fail(parser, f"No se pudo escribir '{args.output}': {e}")
    
    # Resumen de resultados
    line = "=" * 57
    sep = "-" * 57
    unique_clusters = sorted(c for c in set(assignments) if c != -1)
    unclassified = assignments.count(-1)
    print(line)
    print(" RED NEURONAL DE CARPENTER/GROSSBERG - SESIÓN DE APRENDIZAJE")
    print(line)
    print(f" Dataset Fuente     : {args.dataset}")
    print(f" Patrones Aplicados : {len(data)}")
    print(f" Nodos de Entrada F1: {len(headers)} ({', '.join(headers)})")
    print(f" Nivel de Vigilancia: {args.rho}")
    print(line)
    print()
    print(sep)
    print(" RESUMEN DE LA TOPOLOGÍA RESULTANTE EN LA CAPA DE SALIDA (F2)")
    print(sep)
    print(f" Total de Categorías Estabilizadas (M): {len(unique_clusters)}")
    print()
    for c in unique_clusters:
        count = assignments.count(c)
        pct = (count / len(data)) * 100
        proto = [int(v) for v in exemplars[c]]
        print(f" Nodo F2 [{c:02d}] -> {count:3d} instancias ({pct:5.1f}%) | Vector Prototipo Top-Down: {proto}")
    if unclassified:
        print(f"\n Patrones sin clasificar (nodos F2 agotados): {unclassified}")
    if args.output:
        print(f"\n Asignaciones guardadas en: {args.output}")
    print(line)

    # ── Anexo II: Fuzzy C-Means comparativo ─────────────────────────────────
    use_fuzzy = args.fuzzy or (args.fuzzy_c is not None)
    if use_fuzzy:
        n_fcm = args.fuzzy_c if args.fuzzy_c is not None else len(unique_clusters)
        if n_fcm < 2:
            print(f"\n[AVISO --fuzzy] ART-1 estabilizó solo {n_fcm} categoría(s). "
                  "FCM requiere C ≥ 2. Use --fuzzy-c <N> para forzar el número de clusters.")
        else:
            try:
                fcm_hard, fcm_centers, fcm_u = run_fuzzy_cmeans(data, n_fcm)
            except ImportError as exc:
                fail(parser, str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                fail(parser, f"Fuzzy C-Means falló inesperadamente: {exc}")

            from collections import Counter
            fcm_counts: Counter = Counter(fcm_hard)
            art1_labels = sorted(cl for cl in set(assignments) if cl != -1)

            print()
            print(line)
            print(" ANEXO II — FUZZY C-MEANS (Bezdek 1981) — COMPARACIÓN CON ART-1")
            print(f" C={n_fcm} clusters | m=2.0 | error=0.005 | maxiter=1000 | seed=42")
            print(line)
            print()
            print(sep)
            print(" CENTROS FCM Y DISTRIBUCIÓN DE PATRONES")
            print(sep)
            for k in range(n_fcm):
                cnt = fcm_counts.get(k, 0)
                pct = (cnt / len(data)) * 100
                center_bin = [1 if v >= 0.5 else 0 for v in fcm_centers[k]]
                raw_str = "[" + ", ".join(f"{v:.3f}" for v in fcm_centers[k]) + "]"
                print(f" Cluster FCM [{k:02d}] -> {cnt:3d} instancias ({pct:5.1f}%)")
                print(f"   Centro continuo  : {raw_str}")
                print(f"   Centro binarizado: {center_bin}")
                print()

            # Tabla de coincidencia ART-1 × FCM
            print(sep)
            print(" TABLA DE COINCIDENCIA  (filas = ART-1 | columnas = FCM)")
            print(sep)
            col_hdr = ("  ART-1 \\ FCM |" +
                       "".join(f" FCM{k:02d} |" for k in range(n_fcm)) +
                       " TOTAL")
            div = "  " + "-" * (len(col_hdr) - 2)
            print(col_hdr)
            print(div)
            for a_lbl in art1_labels:
                row_counts = [
                    sum(1 for i, a in enumerate(assignments)
                        if a == a_lbl and fcm_hard[i] == k)
                    for k in range(n_fcm)
                ]
                cells = "".join(f"   {cnt:3d} |" for cnt in row_counts)
                print(f"  F2 Nodo [{a_lbl:02d}]  |{cells}   {sum(row_counts):3d}")
            print(div)

            # Pureza estándar: para cada cluster FCM dominancia de etiqueta ART-1
            purity_sum = 0
            for k in range(n_fcm):
                if fcm_counts.get(k, 0) > 0 and art1_labels:
                    best = max(
                        sum(1 for i, a in enumerate(assignments)
                            if a == a_lbl and fcm_hard[i] == k)
                        for a_lbl in art1_labels
                    )
                    purity_sum += best
            purity = (purity_sum / len(data)) * 100 if data else 0.0
            print(f"\n Pureza FCM<->ART-1 (mayoría dominante por cluster FCM): {purity:.1f}%")
            print(" Nota: fracción de patrones asignados a la etiqueta ART-1 más")
            print("       frecuente dentro de cada cluster FCM. 100 % = concordancia total.")
            print(line)

if __name__ == "__main__":
    main()