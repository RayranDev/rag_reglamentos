"""
ingest.py
---------
Script de ingesta completa: procesa todos los PDFs en /data
y los indexa en Qdrant en un solo comando.

Ejecutar desde la raíz del proyecto:
  python ingest.py

Solo necesitas correrlo:
  - La primera vez
  - Cuando agregues PDFs nuevos a /data/

No reprocesa PDFs que ya fueron indexados (reingesta incremental).
"""

import sys
import time
import json
from pathlib import Path

# Asegurar que Python encuentra los módulos en /src
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from console import configurar_salida_utf8
from loader import procesar_pdfs
from chunker    import procesar_textos
from embeddings import generar_embeddings_batch
from vector_db  import (
    obtener_cliente,
    crear_coleccion_si_no_existe,
    crear_indices_payload,
    insertar_chunks,
    obtener_info_coleccion
)

import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")


def main():
    configurar_salida_utf8()
    inicio_total = time.time()

    print()
    print("=" * 60)
    print("  INGESTA DE DOCUMENTOS — RAG REGLAMENTOS")
    print("=" * 60)
    print()

    # ── Paso 1: Cargar y extraer texto de PDFs ────────────────────────────────
    print("PASO 1 — Extracción de texto de PDFs")
    print("─" * 60)
    resultados_loader = procesar_pdfs(
        directorio_data=      str(BASE_DIR / "data"),
        directorio_processed= str(BASE_DIR / "processed")
    )

    pdfs_ok = [r for r in resultados_loader if r["estado"] == "ok"]
    if not pdfs_ok:
        print("\n❌ No hay PDFs procesables en /data")
        print("   Coloca al menos un PDF en la carpeta /data y vuelve a ejecutar.\n")
        sys.exit(1)

    print(f"\n  ✅ {len(pdfs_ok)} PDF(s) procesados correctamente\n")

    # ── Paso 2: Dividir texto en chunks ──────────────────────────────────────
    print("PASO 2 — División en chunks")
    print("─" * 60)
    resumen_chunks = procesar_textos(
        directorio_processed= str(BASE_DIR / "processed"),
        directorio_chunks=    str(BASE_DIR / "chunks")
    )

    total_chunks = sum(r["chunks"] for r in resumen_chunks)
    print(f"\n  ✅ {total_chunks} chunks generados en total\n")

    # ── Paso 3: Generar embeddings ────────────────────────────────────────────
    print("PASO 3 — Generación de embeddings")
    print("─" * 60)
    print(f"  Modelo: {EMBED_MODEL}")
    print("  (Este paso puede tardar varios minutos según la cantidad de PDFs)\n")

    chunks_path = BASE_DIR / "chunks"
    todos_los_chunks = []

    for archivo_json in chunks_path.glob("*.json"):
        with open(archivo_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        todos_los_chunks.extend(chunks)

    if not todos_los_chunks:
        print("❌ No se encontraron chunks para procesar.\n")
        sys.exit(1)

    chunks_con_embeddings = generar_embeddings_batch(
        chunks=        todos_los_chunks,
        modelo=        EMBED_MODEL,
        tamanio_batch= 10
    )

    # ── Paso 4: Indexar en Qdrant ─────────────────────────────────────────────
    print("\nPASO 4 — Indexación en Qdrant")
    print("─" * 60)

    try:
        cliente = obtener_cliente()
        print("  ✅ Conectado a Qdrant\n")
    except Exception as e:
        print(f"  ❌ No se pudo conectar a Qdrant: {e}")
        print("     ¿Está Docker corriendo? Ejecuta: docker-compose up -d\n")
        sys.exit(1)

    crear_coleccion_si_no_existe(cliente)
    crear_indices_payload(cliente)

    resumen_insercion = insertar_chunks(
        cliente= cliente,
        chunks=  chunks_con_embeddings,
        forzar=  False
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    tiempo_total = time.time() - inicio_total
    info         = obtener_info_coleccion(cliente)

    print()
    print("=" * 60)
    print("  INGESTA COMPLETADA")
    print("=" * 60)
    print(f"  PDFs procesados:      {len(pdfs_ok)}")
    print(f"  Chunks generados:     {total_chunks}")
    print(f"  Vectores insertados:  {resumen_insercion['insertados']}")
    print(f"  Vectores omitidos:    {resumen_insercion['omitidos']} (ya indexados)")
    print(f"  Total en Qdrant:      {info['total_vectores']}")
    print(f"  Tiempo total:         {tiempo_total:.1f} segundos")
    print()
    print("  Ahora puedes ejecutar: python src/main.py")
    print()


if __name__ == "__main__":
    main()
