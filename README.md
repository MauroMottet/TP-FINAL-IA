# CarGross: red de Carpenter/Grossberg (ART-1)

Trabajo Final Integrador **"Museo de Redes Neuronales"** · Inteligencia Artificial
Licenciatura en Sistemas de Información · FCyT UADER · IDTI Lab

Implementación en Python de la red de Carpenter/Grossberg descripta por Lippmann (1987, Box 3)
para agrupar sin supervisión los estados operativos de un servidor a partir de telemetría binarizada.
La red crea una categoría nueva (nodo F2) cada vez que un patrón no se parece lo suficiente
a los prototipos aprendidos; el parámetro de vigilancia ρ define qué es "suficiente".

**Integrantes:** Mauro Mottet (líder, código), Agustín Scheffler (arquitectura), Candela Rojas (test),
Florencia Colazo (documentación).
**Docentes:** Dra. Daniela López De Luise, Alejandro Pérez.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `CarGross.py` | Red de Carpenter/Grossberg (programa principal) |
| `dataset_telemetria_base.csv` | Dataset 1: régimen nominal, 100 registros × 5 variables binarias |
| `dataset_telemetria_estres.csv` | Dataset 2: régimen de estrés, 100 registros × 5 variables binarias |
| `metadatos_variables.csv` | Metadatos: origen, umbral y regla de cada variable binaria |
| `preprocesamiento_art1.py` | Genera los datasets binarios a partir de la telemetría cruda |
| `machine_usage_days_1_to_8_grouped_300_seconds.csv` | Telemetría cruda (2243 registros cada 300 s, 5 métricas continuas) |
| `correr_experimentos.py` | Corridas del informe; escribe los resultados en `resultados/` |
| `resultados/` | Resultados de las corridas en CSV |
| `test_cargross.py` | Pruebas automáticas |

## Requisitos

- Python 3.8 o superior.
- No hay dependencias externas: se usa solo la biblioteca estándar (`requirements.txt` está vacío a propósito).

## Instalación

```bash
python3 --version                      # debe ser 3.8 o superior
git clone https://github.com/MauroMottet/TP-FINAL-IA.git
cd TP-FINAL-IA
python3 CarGross.py --help             # muestra el manual (man) del programa
```

En Windows usar `python` en lugar de `python3`.

## Uso

```bash
python3 CarGross.py -d <archivo.csv> [-p RHO] [-m MAX_NODOS] [-o SALIDA.csv] [-v]
```

| Opción | Descripción | Por defecto |
|---|---|---|
| `-d`, `--dataset` | CSV de entrada: encabezado + filas de 0/1 separadas por coma | obligatorio |
| `-p`, `--rho` | Vigilancia ρ, con 0.0 ≤ ρ < 1.0. Más alto = categorías más estrictas | 0.5 |
| `-m`, `--max-nodos` | Cantidad M de nodos F2 disponibles | uno por patrón |
| `-o`, `--output` | Guarda la asignación `fila,cluster` en un CSV (`-1` = sin clasificar) | no se guarda |
| `-v`, `--verbose` | Muestra paso a paso a qué clúster va cada patrón | desactivado |

Ante cualquier corrida infructuosa (archivo inexistente, valores no binarios, filas vacías o
irregulares, ρ fuera de rango, etc.) el programa explica el error, muestra el manual y
termina con código 1.

## Test demo

```bash
python3 CarGross.py -d dataset_telemetria_base.csv -p 0.5
```

Salida esperada:

```
=========================================================
 RED NEURONAL DE CARPENTER/GROSSBERG - SESIÓN DE APRENDIZAJE
=========================================================
 Dataset Fuente     : dataset_telemetria_base.csv
 Patrones Aplicados : 100
 Nodos de Entrada F1: 5 (cpu_alta, ram_alta, disco_saturado, red_in_alta, red_out_alta)
 Nivel de Vigilancia: 0.5
=========================================================

---------------------------------------------------------
 RESUMEN DE LA TOPOLOGÍA RESULTANTE EN LA CAPA DE SALIDA (F2)
---------------------------------------------------------
 Total de Categorías Estabilizadas (M): 8

 Nodo F2 [00] ->  23 instancias ( 23.0%) | Vector Prototipo Top-Down: [0, 0, 1, 0, 0]
 Nodo F2 [01] ->  21 instancias ( 21.0%) | Vector Prototipo Top-Down: [1, 0, 0, 0, 0]
 Nodo F2 [02] ->  18 instancias ( 18.0%) | Vector Prototipo Top-Down: [1, 0, 1, 0, 0]
 Nodo F2 [03] ->  12 instancias ( 12.0%) | Vector Prototipo Top-Down: [0, 1, 0, 0, 0]
 Nodo F2 [04] ->   3 instancias (  3.0%) | Vector Prototipo Top-Down: [1, 1, 0, 0, 0]
 Nodo F2 [05] ->   1 instancias (  1.0%) | Vector Prototipo Top-Down: [0, 1, 1, 0, 0]
 Nodo F2 [06] ->   1 instancias (  1.0%) | Vector Prototipo Top-Down: [0, 0, 1, 0, 1]
 Nodo F2 [07] ->  21 instancias ( 21.0%) | Vector Prototipo Top-Down: [0, 0, 0, 1, 1]
=========================================================
```

Con el dataset de estrés:

```bash
python3 CarGross.py -d dataset_telemetria_estres.csv -p 0.5
```

```
 Total de Categorías Estabilizadas (M): 4

 Nodo F2 [00] ->  19 instancias ( 19.0%) | Vector Prototipo Top-Down: [1, 1, 1, 0, 0]
 Nodo F2 [01] ->  12 instancias ( 12.0%) | Vector Prototipo Top-Down: [0, 0, 0, 1, 1]
 Nodo F2 [02] ->  57 instancias ( 57.0%) | Vector Prototipo Top-Down: [0, 1, 0, 1, 1]
 Nodo F2 [03] ->  12 instancias ( 12.0%) | Vector Prototipo Top-Down: [1, 0, 1, 1, 1]
```

## Datasets

Las 5 variables binarias valen 1 cuando la métrica cruda supera su percentil 90 en los 8 días
(ver `metadatos_variables.csv`):

| Variable | Métrica cruda | Umbral P90 |
|---|---|---|
| `cpu_alta` | `cpu_util_percent` | 53.51 |
| `ram_alta` | `mem_util_percent` | 91.15 |
| `disco_saturado` | `disk_io_percent` | 12.59 |
| `red_in_alta` | `net_in` | 46.11 |
| `red_out_alta` | `net_out` | 36.61 |

- **Base (nominal):** 100 registros con 1 o 2 alertas simultáneas, muestreados en los 8 días (semilla 42) y en orden temporal.
- **Estrés:** los primeros 100 registros con 3 o más alertas simultáneas.
- Los registros sin ninguna alerta (72.85 %) se descartan: con el vector nulo el test de vigilancia
  ‖T·X‖/‖X‖ divide por cero.

Para regenerarlos (el resultado es idéntico byte a byte):

```bash
python3 preprocesamiento_art1.py
```

## Pruebas y experimentos

```bash
python3 -m unittest -v          # 16 pruebas: ejemplo C/E/F de Lippmann, equivalencia con el Box 3, errores, CLI
python3 correr_experimentos.py  # sensibilidad a ρ, evolución de M, efecto del orden, base → estrés
```

## Limitaciones

- Solo admite entradas binarias y no admite vectores nulos.
- El resultado depende del orden de presentación de los patrones (algoritmo del líder).
- El aprendizaje por AND es sensible al ruido: un bit apagado borra ese bit del prototipo para siempre.
- Con ρ alto la cantidad de categorías crece rápidamente; con ρ bajo se mezclan patrones distintos.

## Referencias

1. Lippmann, R. P. (1987). An Introduction to Computing with Neural Nets. *IEEE ASSP Magazine*, 4(2), 4–22.
2. Sánchez-Sinencio, E., & Lau, C. (Eds.). (1992). *Artificial Neural Networks: Paradigms, Applications, and Hardware Implementations*. IEEE Press.
3. Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37(1), 54–115.
