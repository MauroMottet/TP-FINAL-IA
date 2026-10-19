"""
test_cargross.py - Pruebas automáticas de CarGross.py y preprocesamiento_art1.py
Solo usa la biblioteca estándar. Ejecutar desde la carpeta del proyecto con:

    python3 -m unittest -v
"""

import csv
import os
import random
import subprocess
import sys
import tempfile
import unittest

import CarGross as cg

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "dataset_telemetria_base.csv")
ESTRES = os.path.join(HERE, "dataset_telemetria_estres.csv")


def letra(*filas: str):
    """Convierte un dibujo de 5x5 ('1' = píxel negro) en un vector binario de 25 elementos."""
    return [int(c) for fila in filas for c in fila]


# Letras al estilo de la Fig. 11 de Lippmann (1987): C está contenida en E,
# y F comparte la barra superior e izquierda con ambas.
C = letra("11111", "10000", "10000", "10000", "11111")
E = letra("11111", "10000", "11100", "10000", "11111")
F = letra("11111", "10000", "11111", "10000", "10000")
F_RUIDOSA = letra("11110", "10000", "11111", "10000", "10000")  # falta un píxel del borde superior


def box3_literal(data, rho, M):
    """Box 3 de Lippmann tal cual: M nodos con t=1 y b=1/(1+N) compitiendo desde el inicio."""
    N = len(data[0])
    t = [[1.0] * N for _ in range(M)]
    b = [[1.0 / (1 + N)] * N for _ in range(M)]
    asignaciones = []
    for x in data:
        mu = [sum(b[j][i] * x[i] for i in range(N)) for j in range(M)]
        deshabilitados, asignado = set(), -1
        while len(deshabilitados) < M:
            j = max((k for k in range(M) if k not in deshabilitados), key=lambda k: mu[k])
            if sum(t[j][i] * x[i] for i in range(N)) / sum(x) > rho:
                t[j] = [t[j][i] * x[i] for i in range(N)]
                b[j] = [v / (0.5 + sum(t[j])) for v in t[j]]
                asignado = j
                break
            deshabilitados.add(j)
        asignaciones.append(asignado)
    return asignaciones


def tamanios(asignaciones):
    """Cantidad de patrones por clúster, en orden de número de nodo."""
    return [asignaciones.count(c) for c in sorted(set(asignaciones)) if c != -1]


class TestEjemploLippmann(unittest.TestCase):
    """Comportamiento descripto en Lippmann (1987), Fig. 11."""

    def test_vigilancia_alta_crea_un_ejemplar_por_letra(self):
        asig, t = cg.run_carpenter_grossberg([C, E, F], rho=0.9)
        self.assertEqual(asig, [0, 1, 2])
        self.assertEqual(t, [C, E, F])

    def test_vigilancia_menor_fusiona_C_y_E(self):
        # "only two exemplars would have been present after the F; one for F and
        #  one for both C and E that would have been identical to C"
        asig, t = cg.run_carpenter_grossberg([C, E, F], rho=0.8)
        self.assertEqual(asig, [0, 0, 1])
        self.assertEqual(t, [C, F])

    def test_F_ruidosa_degrada_el_ejemplar_por_AND(self):
        asig, t = cg.run_carpenter_grossberg([C, E, F, F_RUIDOSA], rho=0.9)
        self.assertEqual(asig[3], 2)
        self.assertEqual(t[2], F_RUIDOSA)  # F AND F_ruidosa = F_ruidosa


class TestAlgoritmo(unittest.TestCase):

    def test_coincide_con_box3_literal_en_los_datasets(self):
        for ruta in (BASE, ESTRES):
            _, data = cg.load_and_validate_csv(ruta)
            for rho in (0.0, 0.3, 0.5, 0.7, 0.9, 0.99):
                for M in (2, 5, len(data)):
                    with self.subTest(dataset=os.path.basename(ruta), rho=rho, M=M):
                        asig, _ = cg.run_carpenter_grossberg(data, rho, max_nodes=M)
                        self.assertEqual(asig, box3_literal(data, rho, M))

    def test_coincide_con_box3_literal_en_datos_aleatorios(self):
        rnd = random.Random(7)
        for _ in range(100):
            N = rnd.randint(2, 8)
            data = [[rnd.randint(0, 1) for _ in range(N)] for _ in range(30)]
            data = [x for x in data if sum(x) > 0]
            rho, M = rnd.random() * 0.99, rnd.choice([1, 3, len(data)])
            asig, _ = cg.run_carpenter_grossberg(data, rho, max_nodes=M)
            self.assertEqual(asig, box3_literal(data, rho, M))

    def test_nodos_agotados_quedan_sin_clasificar(self):
        asig, t = cg.run_carpenter_grossberg([C, E, F], rho=0.9, max_nodes=2)
        self.assertEqual(asig, [0, 1, -1])
        self.assertEqual(len(t), 2)

    def test_resultados_dataset_base(self):
        _, data = cg.load_and_validate_csv(BASE)
        esperado = {0.3: [43, 24, 12, 21], 0.5: [23, 21, 18, 12, 3, 1, 1, 21]}
        for rho, conteos in esperado.items():
            with self.subTest(rho=rho):
                asig, _ = cg.run_carpenter_grossberg(data, rho)
                self.assertEqual(tamanios(asig), conteos)

    def test_resultados_dataset_estres(self):
        _, data = cg.load_and_validate_csv(ESTRES)
        esperado = {0.3: [21, 79], 0.5: [19, 12, 57, 12], 0.7: [17, 59, 15, 7, 2],
                    0.9: [17, 8, 6, 49, 11, 5, 4]}
        for rho, conteos in esperado.items():
            with self.subTest(rho=rho):
                asig, _ = cg.run_carpenter_grossberg(data, rho)
                self.assertEqual(tamanios(asig), conteos)


class TestCargaCSV(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def csv_temporal(self, contenido: bytes) -> str:
        ruta = os.path.join(self.tmp.name, "datos.csv")
        with open(ruta, "wb") as f:
            f.write(contenido)
        return ruta

    def test_datasets_entregados_son_validos(self):
        for ruta in (BASE, ESTRES):
            headers, data = cg.load_and_validate_csv(ruta)
            self.assertGreaterEqual(len(headers), 5)
            self.assertGreaterEqual(len(data), 50)
            # Ninguna variable es constante: las 5 aportan información
            for i in range(len(headers)):
                self.assertEqual({x[i] for x in data}, {0, 1}, headers[i])

    def test_acepta_bom_de_excel(self):
        headers, data = cg.load_and_validate_csv(self.csv_temporal(b"\xef\xbb\xbfa,b\n1,0\n"))
        self.assertEqual(headers, ["a", "b"])
        self.assertEqual(data, [[1, 0]])

    def test_errores(self):
        casos = {
            "vacío": b"",
            "valor no binario": b"a,b\n1,2\n",
            "vector nulo": b"a,b\n0,0\n",
            "fila irregular": b"a,b,c\n1,0,1\n1,0\n",
            "separador punto y coma": b"a;b\n1;0\n",
            "sin datos": b"a,b\n",
            "encabezado incompleto": b"a,,c\n1,0,1\n",
        }
        for nombre, contenido in casos.items():
            with self.subTest(caso=nombre):
                with self.assertRaises(ValueError):
                    cg.load_and_validate_csv(self.csv_temporal(contenido))

    def test_archivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            cg.load_and_validate_csv(os.path.join(self.tmp.name, "no_existe.csv"))


class TestLineaDeComandos(unittest.TestCase):

    def correr(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, "CarGross.py"), *args],
                              capture_output=True, text=True, cwd=HERE)

    def test_demo(self):
        r = self.correr("-d", "dataset_telemetria_base.csv", "-p", "0.5")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Total de Categorías Estabilizadas (M): 8", r.stdout)

    def test_corridas_infructuosas_muestran_el_man(self):
        casos = [
            [],
            ["-d", "no_existe.csv"],
            ["-d", "dataset_telemetria_base.csv", "-p", "1.0"],
            ["-d", "dataset_telemetria_base.csv", "-p", "abc"],
            ["-d", "dataset_telemetria_base.csv", "-m", "0"],
            ["-d", "machine_usage_days_1_to_8_grouped_300_seconds.csv"],
        ]
        for args in casos:
            with self.subTest(args=args):
                r = self.correr(*args)
                self.assertEqual(r.returncode, 1)
                self.assertIn("EJEMPLOS DE USO", r.stdout)
                self.assertNotIn("Traceback", r.stdout + r.stderr)

    def test_exporta_asignaciones(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida = os.path.join(tmp, "asignaciones.csv")
            r = self.correr("-d", "dataset_telemetria_estres.csv", "-p", "0.3", "-o", salida)
            self.assertEqual(r.returncode, 0)
            with open(salida, encoding="utf-8") as f:
                filas = list(csv.reader(f))
        self.assertEqual(filas[0], ["fila", "cluster"])
        self.assertEqual(len(filas), 101)


class TestPreprocesamiento(unittest.TestCase):

    def test_regenera_los_datasets_identicos(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, os.path.join(HERE, "preprocesamiento_art1.py"), "-s", tmp],
                               capture_output=True, text=True, cwd=HERE)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for nombre in ("dataset_telemetria_base.csv", "dataset_telemetria_estres.csv",
                           "metadatos_variables.csv"):
                with self.subTest(archivo=nombre):
                    with open(os.path.join(tmp, nombre), "rb") as nuevo, \
                         open(os.path.join(HERE, nombre), "rb") as original:
                        self.assertEqual(nuevo.read(), original.read())


if __name__ == "__main__":
    unittest.main()
