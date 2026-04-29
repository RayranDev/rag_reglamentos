"""
embeddings.py
-------------
Responsabilidad: Convertir chunks de texto en vectores numéricos
usando mxbai-embed-large vía Ollama local.

Incluye retry con backoff exponencial para manejar timeouts de Ollama.
"""

import time
import ollama
from pathlib import Path


# Número de dimensiones del modelo mxbai-embed-large
DIMENSIONES_EMBEDDING = 1024


def generar_embedding(texto: str, modelo: str, max_intentos: int = 3) -> list[float] | None:
    """
    Genera el embedding de un texto con reintentos automáticos.

    Usa backoff exponencial: si falla, espera 2s, luego 4s, luego 8s
    antes de reintentar. Esto maneja timeouts temporales de Ollama.

    Args:
        texto:        Texto a convertir en vector.
        modelo:       Nombre del modelo de embeddings en Ollama.
        max_intentos: Número máximo de intentos antes de rendirse.

    Returns:
        Lista de floats representando el vector, o None si falla.
    """
    for intento in range(1, max_intentos + 1):
        try:
            respuesta = ollama.embeddings(model=modelo, prompt=texto)
            return respuesta["embedding"]

        except Exception as e:
            if intento < max_intentos:
                espera = 2 ** intento  # 2s, 4s, 8s
                print(f"    ⚠️  Intento {intento} fallido. Reintentando en {espera}s... ({e})")
                time.sleep(espera)
            else:
                print(f"    ❌ Error tras {max_intentos} intentos: {e}")
                return None


def generar_embeddings_batch(
    chunks: list[dict],
    modelo: str,
    tamanio_batch: int = 10
) -> list[dict]:
    """
    Genera embeddings para una lista de chunks en batches.

    Procesa los chunks en grupos para no saturar Ollama con
    demasiadas peticiones simultáneas.

    Args:
        chunks:        Lista de dicts con la estructura del chunker.
        modelo:        Nombre del modelo de embeddings en Ollama.
        tamanio_batch: Cuántos chunks procesar antes de mostrar progreso.

    Returns:
        Lista de dicts enriquecidos con el campo "embedding".
        Los chunks que fallaron tendrán embedding = None.
    """
    total         = len(chunks)
    procesados    = 0
    fallidos      = 0
    chunks_result = []

    print(f"  🔢 Generando embeddings para {total} chunks...")
    print(f"     Modelo: {modelo}\n")

    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(
            texto=   chunk["texto"],
            modelo=  modelo
        )

        if embedding is not None:
            chunk_enriquecido = {**chunk, "embedding": embedding}
            procesados += 1
        else:
            chunk_enriquecido = {**chunk, "embedding": None}
            fallidos += 1

        chunks_result.append(chunk_enriquecido)

        # Mostrar progreso cada batch
        if (i + 1) % tamanio_batch == 0 or (i + 1) == total:
            print(f"     Progreso: {i + 1}/{total} chunks procesados...")

    print(f"\n  ✅ Embeddings generados: {procesados}/{total}")
    if fallidos > 0:
        print(f"  ⚠️  Chunks fallidos: {fallidos} (se omitirán en Qdrant)")

    return chunks_result


def validar_embedding(embedding: list[float]) -> bool:
    """
    Valida que un embedding tenga las dimensiones correctas.

    Args:
        embedding: Vector a validar.

    Returns:
        True si el embedding es válido, False si no.
    """
    if not embedding:
        return False
    if len(embedding) != DIMENSIONES_EMBEDDING:
        print(f"  ⚠️  Dimensiones inesperadas: {len(embedding)} (esperado: {DIMENSIONES_EMBEDDING})")
        return False
    return True


if __name__ == "__main__":
    """
    Prueba independiente: genera el embedding de una frase
    y verifica que tenga 1024 dimensiones.
    """
    import os
    from dotenv import load_dotenv

    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / ".env")

    modelo = os.getenv("EMBED_MODEL", "mxbai-embed-large")

    print("🧪 Prueba de embeddings.py")
    print(f"   Modelo: {modelo}\n")

    textos_prueba = [
        "El trabajador tiene derecho a 15 días de vacaciones al año.",
        "Las ausencias deben ser justificadas dentro de las 24 horas.",
        "El horario de trabajo es de 8am a 5pm de lunes a viernes."
    ]

    for texto in textos_prueba:
        print(f"  📝 Texto: {texto[:60]}...")
        embedding = generar_embedding(texto=texto, modelo=modelo)

        if embedding and validar_embedding(embedding):
            print(f"  ✅ Vector generado — {len(embedding)} dimensiones")
            print(f"     Primeros 5 valores: {[round(v, 4) for v in embedding[:5]]}\n")
        else:
            print(f"  ❌ Error generando embedding\n")