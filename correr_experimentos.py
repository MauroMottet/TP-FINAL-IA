"""
correr_experimentos.py - Corridas del informe del TFI con ambos datasets
Reutiliza las funciones de CarGross.py y escribe los resultados en CSV dentro
de la carpeta 'resultados/'. Todo es determinista (semillas fijas).

Uso:
    python3 correr_experimentos.py

Experimentos:
  1. Sensibilidad a la vigilancia: M, tamaños y prototipos para rho en 0.1..0.9.
  2. Evolución de M: cantidad de categorías a medida que se presentan los patrones.
  3. Efecto del orden: M y tamaños de clústeres con 20 órdenes aleatorios de
     presentación (semillas 0..19), para rho = 0.3 y 0.5.
  4. Base y luego estrés: se presentan los 100 patrones nominales seguidos de los
     100 de estrés y se mide la pureza respecto del dataset de origen y cuántos
     patrones de estrés caen en categorías creadas durante el régimen nominal.

Solo requiere la biblioteca estándar de Python (>= 3.8).
"""

import csv
import os
import random
from collections import Counter

from CarGross import load_and_validate_csv, run_carpenter_grossberg

DATASETS = {
    "base": "dataset_telemetria_base.csv",
    "estres": "dataset_telemetria_estres.csv",
}
RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]
RHO_REFERENCIA = 0.5
RHOS_ORDEN = [0.3, 0.5]
ORDENES_ALEATORIOS = 20
CARPETA = "resultados"


def escribir_csv(nombre, encabezado, filas):
    ruta = os.path.join(CARPETA, nombre)
    with open(ruta, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(encabezado)
        writer.writerows(filas)
    return ruta


def prototipo_str(t):
    return "".join(str(int(v)) for v in t)


def tamanios(asignaciones):
    conteo = Counter(c for c in asignaciones if c != -1)
    return [conteo[c] for c in sorted(conteo)]


def experimento_vigilancia(datos):
    filas = []
    print("\n1. SENSIBILIDAD A LA VIGILANCIA")
    print(f"   {'dataset':<8} {'rho':>4} {'M':>3}  tamaños de los clústeres")
    for nombre, data in datos.items():
        for rho in RHOS:
            asig, t = run_carpenter_grossberg(data, rho)
            for c, n in enumerate(tamanios(asig)):
                filas.append([nombre, rho, len(t), c, n, prototipo_str(t[c])])
            print(f"   {nombre:<8} {rho:>4} {len(t):>3}  {tamanios(asig)}")
    return escribir_csv("vigilancia.csv", ["dataset", "rho", "M", "cluster", "instancias", "prototipo"], filas)


def experimento_evolucion(datos):
    filas = []
    print(f"\n2. EVOLUCIÓN DE M (rho = {RHO_REFERENCIA})")
    for nombre, data in datos.items():
        asig, _ = run_carpenter_grossberg(data, RHO_REFERENCIA)
        maximo = -1
        hitos = []
        for paso, c in enumerate(asig, start=1):
            maximo = max(maximo, c)
            filas.append([nombre, paso, maximo + 1])
            if c == maximo and asig.index(c) == paso - 1:
                hitos.append(paso)
        print(f"   {nombre:<8} nuevas categorías en los patrones {hitos}")
    return escribir_csv("evolucion_M.csv", ["dataset", "patron", "M"], filas)


def experimento_orden(datos):
    filas = []
    print(f"\n3. EFECTO DEL ORDEN DE PRESENTACIÓN ({ORDENES_ALEATORIOS} órdenes aleatorios)")
    print(f"   {'dataset':<8} {'rho':>4}  M observado -> veces        particiones distintas")
    for nombre, data in datos.items():
        for rho in RHOS_ORDEN:
            valores_M = []
            particiones = set()
            for semilla in range(ORDENES_ALEATORIOS):
                mezcla = data[:]
                random.Random(semilla).shuffle(mezcla)
                asig, t = run_carpenter_grossberg(mezcla, rho)
                orden_tamanios = tuple(sorted(tamanios(asig), reverse=True))
                valores_M.append(len(t))
                particiones.add(orden_tamanios)
                filas.append([nombre, rho, semilla, len(t), " ".join(map(str, orden_tamanios))])
            distribucion = str(dict(sorted(Counter(valores_M).items())))
            print(f"   {nombre:<8} {rho:>4}  {distribucion:<26} {len(particiones)}/{ORDENES_ALEATORIOS}")
    return escribir_csv("efecto_orden.csv", ["dataset", "rho", "semilla", "M", "tamanios_ordenados"], filas)


def experimento_base_luego_estres(datos):
    data = datos["base"] + datos["estres"]
    origen = ["base"] * len(datos["base"]) + ["estres"] * len(datos["estres"])
    filas = []
    print("\n4. BASE Y LUEGO ESTRÉS (detección de régimen nuevo)")
    print(f"   {'rho':>4} {'M':>3} {'pureza':>7}  estrés en categorías nominales")
    for rho in RHOS:
        asig, t = run_carpenter_grossberg(data, rho)
        categorias_nominales = set(asig[:len(datos["base"])])
        por_cluster = {}
        for c, o in zip(asig, origen):
            por_cluster.setdefault(c, Counter())[o] += 1
        pureza = sum(max(cnt.values()) for cnt in por_cluster.values()) / len(data)
        estres_absorbido = sum(1 for c in asig[len(datos["base"]):] if c in categorias_nominales)
        filas.append([rho, len(t), f"{pureza:.3f}", estres_absorbido])
        print(f"   {rho:>4} {len(t):>3} {pureza:>7.3f}  {estres_absorbido}/{len(datos['estres'])}")
    return escribir_csv("base_luego_estres.csv", ["rho", "M", "pureza", "estres_en_categorias_nominales"], filas)


def main():
    os.makedirs(CARPETA, exist_ok=True)
    datos = {nombre: load_and_validate_csv(ruta)[1] for nombre, ruta in DATASETS.items()}
    archivos = [
        experimento_vigilancia(datos),
        experimento_evolucion(datos),
        experimento_orden(datos),
        experimento_base_luego_estres(datos),
    ]
    print("\nResultados guardados en:")
    for ruta in archivos:
        print(f"  [OK] {ruta}")


if __name__ == "__main__":
    main()
