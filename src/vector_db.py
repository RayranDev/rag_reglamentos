"""
vector_db.py
------------
Responsabilidad: Gestionar la colección en Qdrant.
Crear colección, insertar vectores, evitar duplicados
y soportar reingesta incremental.

Cambios v2:
  - Incluye `tipo_doc` y `articulo` en el payload de cada punto
  - Crea índices de payload para tipo_doc, fuente y articulo
    (necesarios para que los filtros del retriever sean eficientes)
  - Nueva función `eliminar_fuente()` para reindexar un documento
    sin borrar toda la colección
"""

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QDRANT_URL      = os.getenv("QDRANT_URL",      "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reglamentos")
DIMENSIONES     = 1024


# ─── Cliente ──────────────────────────────────────────────────────────────────

def obtener_cliente() -> QdrantClient:
    """
    Retorna un cliente conectado a Qdrant.

    Returns:
        Instancia de QdrantClient.
    """
    return QdrantClient(url=QDRANT_URL)


# ─── Colección ────────────────────────────────────────────────────────────────

def crear_coleccion_si_no_existe(cliente: QdrantClient) -> None:
    """
    Crea la colección en Qdrant si no existe (idempotente).

    Args:
        cliente: Instancia activa de QdrantClient.
    """
    existentes = [c.name for c in cliente.get_collections().collections]

    if COLLECTION_NAME not in existentes:
        cliente.create_collection(
            collection_name= COLLECTION_NAME,
            vectors_config=  VectorParams(
                size=     DIMENSIONES,
                distance= Distance.COSINE
            )
        )
        print(f"  ✅ Colección '{COLLECTION_NAME}' creada en Qdrant")
    else:
        print(f"  ℹ️  Colección '{COLLECTION_NAME}' ya existe — se reutiliza")


def crear_indices_payload(cliente: QdrantClient) -> None:
    """
    Crea índices en los campos de filtrado más usados.

    Sin estos índices, Qdrant hace full-scan en cada búsqueda filtrada,
    lo que es lento cuando la colección crece.

    Campos indexados:
      - tipo_doc:  Para filtrar RIT vs BPM en el retriever
      - fuente:    Para la reingesta incremental (fuente_ya_indexada)
      - articulo:  Para búsquedas futuras por artículo específico

    Operación idempotente: si el índice ya existe, no falla.

    Args:
        cliente: Instancia activa de QdrantClient.
    """
    campos = ["tipo_doc", "fuente", "articulo"]

    for campo in campos:
        try:
            cliente.create_payload_index(
                collection_name= COLLECTION_NAME,
                field_name=      campo,
                field_schema=    PayloadSchemaType.KEYWORD
            )
            print(f"  ✅ Índice creado: {campo}")
        except Exception as e:
            # El índice ya existe o hubo un error menor — continuar
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                print(f"  ℹ️  Índice ya existe: {campo}")
            else:
                print(f"  ⚠️  Error creando índice '{campo}': {e}")


# ─── Verificación de fuente ───────────────────────────────────────────────────

def fuente_ya_indexada(cliente: QdrantClient, nombre_fuente: str) -> bool:
    """
    Verifica si un PDF ya fue indexado en la colección.

    Permite la reingesta incremental: solo procesar archivos nuevos.

    Args:
        cliente:        Instancia activa de QdrantClient.
        nombre_fuente:  Nombre del archivo PDF (ej: "RIT_PLASTITEC.pdf").

    Returns:
        True si la fuente ya está indexada, False si no.
    """
    resultados = cliente.scroll(
        collection_name= COLLECTION_NAME,
        scroll_filter=   Filter(
            must=[FieldCondition(
                key=   "fuente",
                match= MatchValue(value=nombre_fuente)
            )]
        ),
        limit= 1
    )
    return len(resultados[0]) > 0


def eliminar_fuente(cliente: QdrantClient, nombre_fuente: str) -> int:
    """
    Elimina todos los puntos de una fuente específica de Qdrant.

    Útil cuando se necesita reindexar un documento sin borrar
    toda la colección.

    Args:
        cliente:        Instancia activa de QdrantClient.
        nombre_fuente:  Nombre del archivo PDF a eliminar.

    Returns:
        Número de puntos eliminados (aproximado).
    """
    from qdrant_client.models import FilterSelector

    try:
        cliente.delete(
            collection_name= COLLECTION_NAME,
            points_selector= FilterSelector(
                filter=Filter(
                    must=[FieldCondition(
                        key=   "fuente",
                        match= MatchValue(value=nombre_fuente)
                    )]
                )
            )
        )
        print(f"  🗑️  Puntos de '{nombre_fuente}' eliminados de Qdrant")
        return 0   # Qdrant no retorna el conteo exacto en delete
    except Exception as e:
        print(f"  ❌ Error eliminando fuente '{nombre_fuente}': {e}")
        return -1


# ─── Inserción de chunks ──────────────────────────────────────────────────────

def insertar_chunks(
    cliente: QdrantClient,
    chunks:  list[dict],
    forzar:  bool = False
) -> dict:
    """
    Inserta chunks con embeddings en Qdrant.

    Payload almacenado por punto:
      - chunk_id:  ID único del chunk
      - fuente:    Nombre del PDF de origen
      - pagina:    Número de página
      - texto:     Contenido del chunk
      - tipo_doc:  "RIT", "BPM" u "otro"  ← NUEVO
      - articulo:  Artículo más cercano    ← NUEVO

    Soporta reingesta incremental: si la fuente ya está en Qdrant
    y forzar=False, se omite. Si forzar=True, primero la elimina
    y luego la reindexaa.

    Args:
        cliente:  Instancia activa de QdrantClient.
        chunks:   Lista de chunks del chunker (con campo "embedding").
        forzar:   Si True, reindexar aunque la fuente ya exista.

    Returns:
        Dict con resumen: {insertados, omitidos, fallidos}
    """
    insertados = 0
    omitidos   = 0
    fallidos   = 0

    # Agrupar chunks por fuente
    fuentes: dict[str, list] = {}
    for chunk in chunks:
        fuente = chunk.get("fuente", "desconocido")
        fuentes.setdefault(fuente, []).append(chunk)

    for fuente, chunks_fuente in fuentes.items():
        print(f"\n  📎 Procesando fuente: {fuente}")

        tipo_doc_fuente = chunks_fuente[0].get("tipo_doc", "otro") if chunks_fuente else "otro"
        print(f"     Tipo: {tipo_doc_fuente}")

        if not forzar and fuente_ya_indexada(cliente, fuente):
            print(f"  ⏭️  Ya indexada — se omite (usa forzar=True para reindexar)")
            omitidos += len(chunks_fuente)
            continue

        # Si forzar=True, eliminar los puntos existentes primero
        if forzar and fuente_ya_indexada(cliente, fuente):
            print(f"  🔄 Reindexando — eliminando versión anterior...")
            eliminar_fuente(cliente, fuente)

        # Construir puntos para Qdrant
        puntos = []
        for chunk in chunks_fuente:
            if chunk.get("embedding") is None:
                fallidos += 1
                continue

            punto = PointStruct(
                id=      str(uuid.uuid4()),
                vector=  chunk["embedding"],
                payload= {
                    "chunk_id": chunk["chunk_id"],
                    "fuente":   chunk["fuente"],
                    "pagina":   chunk["pagina"],
                    "texto":    chunk["texto"],
                    "tipo_doc": chunk.get("tipo_doc", "otro"),    # NUEVO
                    "articulo": chunk.get("articulo", ""),         # NUEVO
                }
            )
            puntos.append(punto)

        if not puntos:
            print(f"  ⚠️  Sin puntos válidos para '{fuente}'")
            continue

        # Insertar en lotes de 100
        tamanio_lote = 100
        for i in range(0, len(puntos), tamanio_lote):
            lote = puntos[i:i + tamanio_lote]
            cliente.upsert(
                collection_name= COLLECTION_NAME,
                points=          lote
            )

        insertados += len(puntos)
        print(f"  ✅ {len(puntos)} chunks insertados (tipo: {tipo_doc_fuente})")

    return {
        "insertados": insertados,
        "omitidos":   omitidos,
        "fallidos":   fallidos
    }


# ─── Info de colección ────────────────────────────────────────────────────────

def obtener_info_coleccion(cliente: QdrantClient) -> dict:
    """
    Retorna información básica sobre la colección.

    Args:
        cliente: Instancia activa de QdrantClient.

    Returns:
        Dict con nombre, total de vectores y estado.
    """
    info = cliente.get_collection(COLLECTION_NAME)
    return {
        "nombre":         COLLECTION_NAME,
        "total_vectores": info.points_count,
        "estado":         info.status
    }


if __name__ == "__main__":
    print("🧪 Prueba de vector_db.py\n")

    cliente = obtener_cliente()
    print(f"  ✅ Conectado a Qdrant en {QDRANT_URL}\n")

    crear_coleccion_si_no_existe(cliente)
    print()
    crear_indices_payload(cliente)

    info = obtener_info_coleccion(cliente)
    print(f"\n  📊 Colección: {info['nombre']}")
    print(f"     Vectores:  {info['total_vectores']}")
    print(f"     Estado:    {info['estado']}")