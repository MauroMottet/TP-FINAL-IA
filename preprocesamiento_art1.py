"""
Script de Preprocesamiento y Binarización de Telemetría para Redes ART1 (Carpenter-Grossberg)
Trabajo Final Integrador - Inteligencia Artificial

Genera:
  - dataset_telemetria_base.csv
  - dataset_telemetria_estres.csv
  - metadatos_variables.csv  (umbral y significado de cada variable binaria)

Uso:
    python3 preprocesamiento_art1.py
    python3 preprocesamiento_art1.py -i otro_archivo.csv -s carpeta_salida

Solo requiere la biblioteca estándar de Python (>= 3.8).
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

SEMILLA_MUESTREO = 42  # fija para que el dataset base sea reproducible

def calcular_percentil_90_lineal(valores):
    """
    Calcula el Percentil 90 empírico usando interpolación lineal
    (equivalente al estándar de numpy/pandas quantile).
    """
    vals = sorted(valores)
    n = len(vals)
    idx = (n - 1) * 0.90
    i = int(idx)
    frac = idx - i
    if i + 1 < n:
        return vals[i] + frac * (vals[i + 1] - vals[i])
    return vals[i]

def parse_arguments():
    """Argumentos opcionales: los valores por defecto reproducen los datasets del repositorio."""
    parser = argparse.ArgumentParser(
        prog="preprocesamiento_art1.py",
        description="Binariza la telemetría cruda (umbral P90) y genera los datasets para CarGross.py."
    )
    parser.add_argument("-i", "--entrada", default="machine_usage_days_1_to_8_grouped_300_seconds.csv",
                        help="CSV de telemetría cruda. Por defecto: %(default)s")
    parser.add_argument("-s", "--salida-dir", default=".",
                        help="Carpeta donde se escriben los CSV generados. Por defecto: carpeta actual")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_file = args.entrada
    if not os.path.isfile(input_file):
        print(f"[ERROR] No se encontró el archivo: {input_file}")
        sys.exit(1)
    os.makedirs(args.salida_dir, exist_ok=True)

    # 1. Carga del dataset original
    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = [row for row in reader if any(row.values())]

    if not raw_rows:
        print(f"[ERROR] El archivo {input_file} no contiene registros.")
        sys.exit(1)

    total_registros_raw = len(raw_rows)

    columnas_fuente = [
        "cpu_util_percent",
        "mem_util_percent",
        "disk_io_percent",
        "net_in",
        "net_out"
    ]
    
    mapeo_columnas = {
        "cpu_util_percent": "cpu_alta",
        "mem_util_percent": "ram_alta",
        "disk_io_percent": "disco_saturado",
        "net_in": "red_in_alta",
        "net_out": "red_out_alta"
    }

    columnas_binarias = [mapeo_columnas[c] for c in columnas_fuente]

    faltantes = [c for c in columnas_fuente if c not in raw_rows[0]]
    if faltantes:
        print(f"[ERROR] Faltan columnas en {input_file}: {', '.join(faltantes)}")
        sys.exit(1)

    # 2. Cálculo de umbrales objetivos P90
    umbrales = {}
    for col in columnas_fuente:
        try:
            valores = [float(row[col]) for row in raw_rows]
        except ValueError as e:
            print(f"[ERROR] Valor no numérico en la columna {col}: {e}")
            sys.exit(1)
        umbrales[col] = calcular_percentil_90_lineal(valores)

    # 3. Binarización de las métricas
    binarizados = []
    for idx, row in enumerate(raw_rows):
        registro_bin = {
            "cpu_alta": 1 if float(row["cpu_util_percent"]) > umbrales["cpu_util_percent"] else 0,
            "ram_alta": 1 if float(row["mem_util_percent"]) > umbrales["mem_util_percent"] else 0,
            "disco_saturado": 1 if float(row["disk_io_percent"]) > umbrales["disk_io_percent"] else 0,
            "red_in_alta": 1 if float(row["net_in"]) > umbrales["net_in"] else 0,
            "red_out_alta": 1 if float(row["net_out"]) > umbrales["net_out"] else 0,
        }
        registro_bin["_orig_idx"] = idx
        registro_bin["_suma_bits"] = sum(registro_bin[c] for c in columnas_binarias)
        binarizados.append(registro_bin)

    # 4. Tratamiento de restricción matemática ART1 (división por cero)
    # ||X|| = 0 produce indeterminación en el test de vigilancia ||T · X|| / ||X||
    filas_nulas = [r for r in binarizados if r["_suma_bits"] == 0]
    filas_activas = [r for r in binarizados if r["_suma_bits"] > 0]

    cant_nulas = len(filas_nulas)
    pct_nulas = (cant_nulas / total_registros_raw) * 100
    cant_activas = len(filas_activas)
    pct_activas = (cant_activas / total_registros_raw) * 100

    # 5. Selección y generación de los 2 Datasets
    # Dataset 1 (Base): régimen operativo nominal = registros con 1 o 2 alertas simultáneas.
    # Se toma una muestra de 100 repartida en los 8 días (semilla fija -> reproducible) y se
    # conserva el orden temporal. Tomar los primeros 100 registros dejaba red_in_alta y
    # red_out_alta siempre en 0, porque en los días 1 a 3 la red nunca supera su P90.
    filas_nominales = [r for r in filas_activas if r["_suma_bits"] <= 2]
    if len(filas_nominales) < 100:
        print(f"[ERROR] Solo hay {len(filas_nominales)} registros nominales; se necesitan 100.")
        sys.exit(1)
    muestra = random.Random(SEMILLA_MUESTREO).sample(filas_nominales, 100)
    dataset_base = sorted(muestra, key=lambda r: r["_orig_idx"])

    # Dataset 2 (Estrés): Régimen operativo exigido con concurrencia de picos (>= 3 variables simultáneas en 1)
    filas_estres_pool = [r for r in filas_activas if r["_suma_bits"] >= 3]
    dataset_estres = filas_estres_pool[:100]

    # 6. Exportación a CSV limpio (sin índice, delimitado por comas)
    archivo_base = os.path.join(args.salida_dir, "dataset_telemetria_base.csv")
    archivo_estres = os.path.join(args.salida_dir, "dataset_telemetria_estres.csv")
    archivo_metadatos = os.path.join(args.salida_dir, "metadatos_variables.csv")

    def exportar_csv(archivo, datos):
        # lineterminator "\n": salida idéntica byte a byte en Windows, Linux y macOS
        with open(archivo, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas_binarias, lineterminator="\n")
            writer.writeheader()
            for row in datos:
                # Limpieza de metadatos auxiliares
                limpio = {col: int(row[col]) for col in columnas_binarias}
                writer.writerow(limpio)

    exportar_csv(archivo_base, dataset_base)
    exportar_csv(archivo_estres, dataset_estres)

    # Metadatos: documentan de dónde sale cada variable binaria (no los requiere CarGross.py)
    descripciones = {
        "cpu_alta": "Uso de CPU por encima del percentil 90 del período",
        "ram_alta": "Uso de memoria por encima del percentil 90 del período",
        "disco_saturado": "Uso de E/S de disco por encima del percentil 90 del período",
        "red_in_alta": "Tráfico de red entrante por encima del percentil 90 del período",
        "red_out_alta": "Tráfico de red saliente por encima del percentil 90 del período",
    }
    with open(archivo_metadatos, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["variable_binaria", "columna_fuente", "umbral_p90", "regla", "descripcion"])
        for col in columnas_fuente:
            var_bin = mapeo_columnas[col]
            writer.writerow([var_bin, col, f"{umbrales[col]:.6f}",
                             f"1 si {col} > umbral_p90; 0 en caso contrario", descripciones[var_bin]])

    # 7. Informe Final de Métricas y Análisis Estadístico
    print("=" * 80)
    print("INFORME DE PREPROCESAMIENTO Y GENERACIÓN DE DATASETS PARA REDES ART1")
    print("=" * 80)

    print("\n1. UMBRALES OBJETIVOS EMPÍRICOS (PERCENTIL 90):")
    print("-" * 80)
    for col in columnas_fuente:
        var_bin = mapeo_columnas[col]
        print(f"  • {col:<18} -> {var_bin:<16}: {umbrales[col]:.6f}")

    print("\n2. TRATAMIENTO DE LA RESTRICCIÓN MATEMÁTICA DE CARPENTER/GROSSBERG:")
    print("-" * 80)
    print(f"  • Total de registros evaluados (raw):           {total_registros_raw:,}")
    print(f"  • Vectores nulos descartados ([0,0,0,0,0]):     {cant_nulas:,} ({pct_nulas:.2f}%)")
    print(f"  • Vectores con actividad detectable (||X|| >= 1): {cant_activas:,} ({pct_activas:.2f}%)")
    print("  • Justificación matemática: En ART1, el test de vigilancia evalúa ||T · X|| / ||X||.")
    print("    Si X = [0,0,0,0,0], ||X|| = 0 provoca división por cero e indeterminación de categoría.")
    print("    El descarte de períodos pasivos modela el flujo como un log de eventos de telemetría activa.")

    def imprimir_estadisticas_dataset(nombre, archivo, datos):
        print(f"\n{nombre} ({archivo}):")
        print("-" * 80)
        n = len(datos)
        cols = len(columnas_binarias)
        print(f"  • Dimensión final: {n} filas × {cols} columnas (valores estrictamente enteros 0 y 1)")
        
        print("\n  • Frecuencia y porcentaje de 1s y 0s por columna:")
        for col in columnas_binarias:
            ones = sum(r[col] for r in datos)
            zeros = n - ones
            pct_ones = (ones / n) * 100
            pct_zeros = (zeros / n) * 100
            print(f"    - {col:<16}: 1s = {ones:3d} ({pct_ones:5.1f}%) | 0s = {zeros:3d} ({pct_zeros:5.1f}%)")

        patrones = Counter(tuple(r[col] for col in columnas_binarias) for r in datos)
        print(f"\n  • Patrones binarios únicos ({len(patrones)} combinaciones):")
        print(f"    {'Patrón [cpu, ram, disk, net_in, net_out]':<45} | {'Frecuencia':<10} | {'Porcentaje':<10}")
        print("    " + "-" * 72)
        for pat, count in patrones.most_common():
            pct = (count / n) * 100
            pat_str = str(list(pat))
            print(f"    {pat_str:<45} | {count:10d} | {pct:9.1f}%")

    imprimir_estadisticas_dataset("3. DATASET 1: RÉGIMEN BASE / NOMINAL", archivo_base, dataset_base)
    imprimir_estadisticas_dataset("4. DATASET 2: RÉGIMEN OPERATIVO DE ESTRÉS", archivo_estres, dataset_estres)
    print("\n" + "=" * 80)
    print("EXPORTACIÓN COMPLETADA EXITOSAMENTE:")
    print(f"  [OK] {os.path.abspath(archivo_base)}")
    print(f"  [OK] {os.path.abspath(archivo_estres)}")
    print(f"  [OK] {os.path.abspath(archivo_metadatos)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
