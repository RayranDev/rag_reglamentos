import csv
import time
import requests
import os
from pathlib import Path

ARCHIVO_PREGUNTAS = "preguntas_RIT_PLASTITEC__1_.csv"
API_URL = "http://localhost:8000/ask"

def main():
    if not Path(ARCHIVO_PREGUNTAS).exists():
        print(f"Error: No se encuentra {ARCHIVO_PREGUNTAS}")
        return

    preguntas = []
    with open(ARCHIVO_PREGUNTAS, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            preguntas.append(fila)

    limite = int(os.getenv("EVAL_LIMIT", "0"))
    if limite > 0:
        preguntas = preguntas[:limite]

    total = len(preguntas)
    print("=" * 50)
    print(f"EVALUANDO API RAG PLASTITEC")
    print(f"Total preguntas: {total}")
    print("=" * 50)

    confianza_alta = 0
    tiempos = []

    for i, fila in enumerate(preguntas, 1):
        pregunta = fila.get("pregunta", "")
        print(f"[{i:03d}/{total}] {pregunta[:50]}...", end=" ", flush=True)

        try:
            inicio = time.time()
            res = requests.post(API_URL, json={"pregunta": pregunta}, timeout=40)
            res.raise_for_status()
            data = res.json()
            tiempo_req = time.time() - inicio

            confianza = data.get("confianza", "baja")
            if confianza == "alta":
                confianza_alta += 1

            tiempos.append(tiempo_req)
            print(f"-> Confianza: {confianza} | Tiempo: {tiempo_req:.2f}s")
        except Exception as e:
            print(f"-> ERROR: {e}")

    if total > 0:
        pct_alta = (confianza_alta / total) * 100
        promedio_tiempo = sum(tiempos) / len(tiempos) if tiempos else 0
        print("\n" + "=" * 50)
        print("RESULTADOS DE EVALUACIÓN API")
        print(f"Confianza Alta: {confianza_alta}/{total} ({pct_alta:.2f}%)")
        print(f"Tiempo Promedio: {promedio_tiempo:.2f}s")
        print("=" * 50)

if __name__ == "__main__":
    main()
