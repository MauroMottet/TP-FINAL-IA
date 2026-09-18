"""
Script de Preprocesamiento y Binarización de Telemetría para Redes ART1 (Carpenter-Grossberg)
Trabajo Final Integrador - Inteligencia Artificial

Genera:
  - dataset_telemetria_base.csv
  - dataset_telemetria_estres.csv
"""

import csv
import os
from collections import Counter

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

def main():
    input_file = "machine_usage_days_1_to_8_grouped_300_seconds.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encontró el archivo: {input_file}")

    # 1. Carga del dataset original
    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = [row for row in reader if any(row.values())]

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

    # 2. Cálculo de umbrales objetivos P90
    umbrales = {}
    for col in columnas_fuente:
        valores = [float(row[col]) for row in raw_rows]
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
    # Dataset 1 (Base): Secuencia continua de 100 eventos del régimen operativo basal/nominal
    # Se extraen los primeros 100 registros con actividad detectable (Días 1 a 3)
    dataset_base = filas_activas[:100]

    # Dataset 2 (Estrés): Régimen operativo exigido con concurrencia de picos (>= 3 variables simultáneas en 1)
    filas_estres_pool = [r for r in filas_activas if r["_suma_bits"] >= 3]
    dataset_estres = filas_estres_pool[:100]

    # 6. Exportación a CSV limpio (sin índice, delimitado por comas)
    archivo_base = "dataset_telemetria_base.csv"
    archivo_estres = "dataset_telemetria_estres.csv"

    def exportar_csv(archivo, datos):
        with open(archivo, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas_binarias)
            writer.writeheader()
            for row in datos:
                # Limpieza de metadatos auxiliares
                limpio = {col: int(row[col]) for col in columnas_binarias}
                writer.writerow(limpio)

    exportar_csv(archivo_base, dataset_base)
    exportar_csv(archivo_estres, dataset_estres)

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
    print("=" * 80)

if __name__ == "__main__":
    main()
