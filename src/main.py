"""
main.py
-------
Responsabilidad: Orquestar el flujo completo del sistema RAG.

Flujo en tiempo real:
  Input → logger(entrada) → classifier
            ├─ SENSIBLE  → mensaje RRHH → logger(salida)
            └─ PERMITIDA → retriever → llm → respuesta → logger(salida)

Requiere que la ingesta ya haya sido ejecutada (ingest.py).
"""

import time
import os
import sys
import ollama
from pathlib import Path
from dotenv import load_dotenv

# Asegurar que Python encuentra los módulos en /src
sys.path.insert(0, str(Path(__file__).resolve().parent))

from console    import configurar_salida_utf8
from logger     import registrar_consulta, registrar_error, nueva_sesion
from classifier import clasificar_pregunta
from retriever  import buscar_chunks_relevantes, formatear_contexto
from llm        import generar_respuesta, respuesta_no_encontrada


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
LLM_MODEL  = os.getenv("LLM_MODEL",  "llama3:8b")


# ─── Mensaje de respuesta bloqueada ──────────────────────────────────────────

MENSAJE_SENSIBLE = (
    "Para este tipo de consultas, por favor dirígete al área de "
    "Recursos Humanos, donde podrán brindarte una orientación "
    "más personalizada."
)


# ─── Verificaciones de inicio ────────────────────────────────────────────────

def _verificar_ollama() -> bool:
    """
    Verifica que Ollama esté corriendo y el modelo esté disponible.

    Returns:
        True si Ollama responde correctamente, False si no.
    """
    try:
        modelos = ollama.list()
        nombres = [m["name"] for m in modelos.get("models", [])]
        if not any(LLM_MODEL.split(":")[0] in n for n in nombres):
            print(f"  ⚠️  El modelo '{LLM_MODEL}' no está descargado.")
            print(f"     Ejecuta: ollama pull {LLM_MODEL}")
            return False
        return True
    except Exception:
        print("  ❌ Ollama no responde.")
        print("     Asegúrate de que Ollama esté corriendo.")
        print("     En Windows/macOS: abre la app Ollama.")
        print("     En Linux: ejecuta 'ollama serve' en otra terminal.")
        return False


def _verificar_qdrant() -> bool:
    """
    Verifica que Qdrant esté corriendo y accesible.

    Returns:
        True si Qdrant responde correctamente, False si no.
    """
    try:
        from qdrant_client import QdrantClient
        cliente = QdrantClient(url=QDRANT_URL, timeout=3)
        cliente.get_collections()
        return True
    except Exception:
        print("  ❌ Qdrant no responde.")
        print("     Asegúrate de que Docker esté corriendo y ejecuta:")
        print("     docker-compose up -d")
        print(f"     Luego verifica en: {QDRANT_URL}/dashboard")
        return False


def _verificar_datos_indexados() -> bool:
    """
    Verifica que existan vectores indexados en Qdrant.

    Returns:
        True si hay datos, False si la colección está vacía.
    """
    try:
        from qdrant_client import QdrantClient
        cliente = QdrantClient(url=QDRANT_URL, timeout=3)
        colecciones = [c.name for c in cliente.get_collections().collections]

        collection_name = os.getenv("COLLECTION_NAME", "reglamentos")
        if collection_name not in colecciones:
            print(f"  ⚠️  La colección '{collection_name}' no existe en Qdrant.")
            print("     Ejecuta primero: python ingest.py")
            return False

        info = cliente.get_collection(collection_name)
        if info.points_count == 0:
            print("  ⚠️  La colección está vacía. No hay PDFs indexados.")
            print("     Coloca PDFs en /data y ejecuta: python ingest.py")
            return False

        print(f"  ✅ {info.points_count} fragmentos indexados listos para consulta")
        return True
    except Exception:
        return False


# ─── Flujo principal de consulta ─────────────────────────────────────────────

def procesar_pregunta(pregunta: str, session_id: str) -> str:
    """
    Ejecuta el pipeline completo para una pregunta del usuario.

    Flujo:
      1. Clasificar si es sensible o permitida
      2. Si sensible: retornar mensaje RRHH
      3. Si permitida: buscar chunks → generar respuesta con LLM
      4. Registrar todo en el logger

    Args:
        pregunta:   Texto de la pregunta del usuario.
        session_id: ID de sesión para el logger.

    Returns:
        Texto de la respuesta a mostrar al usuario.
    """
    inicio = time.time()

    # ── Paso 1: Clasificar la pregunta ────────────────────────────────────────
    clasificacion = clasificar_pregunta(pregunta)
    resultado_clf = clasificacion["resultado"]
    capa_clf      = clasificacion["capa_usada"]

    # ── Paso 2: Bloquear si es sensible ───────────────────────────────────────
    if resultado_clf == "SENSIBLE":
        tiempo_ms = (time.time() - inicio) * 1000
        registrar_consulta(
            session_id=        session_id,
            pregunta=          pregunta,
            clasificacion=     "SENSIBLE",
            chunks_usados=     None,
            respuesta=         None,
            tiempo_ms=         tiempo_ms,
            capa_clasificador= capa_clf
        )
        return MENSAJE_SENSIBLE

    # ── Paso 3: Buscar chunks relevantes ──────────────────────────────────────
    chunks = buscar_chunks_relevantes(pregunta)

    if not chunks:
        respuesta_dict = respuesta_no_encontrada()
        tiempo_ms      = (time.time() - inicio) * 1000
        registrar_consulta(
            session_id=        session_id,
            pregunta=          pregunta,
            clasificacion=     "PERMITIDA",
            chunks_usados=     [],
            respuesta=         respuesta_dict["respuesta"],
            tiempo_ms=         tiempo_ms,
            capa_clasificador= capa_clf
        )
        return respuesta_dict["respuesta"]

    # ── Paso 4: Generar respuesta con el LLM ──────────────────────────────────
    contexto       = formatear_contexto(chunks)
    respuesta_dict = generar_respuesta(contexto, pregunta)
    tiempo_ms      = (time.time() - inicio) * 1000

    # ── Paso 5: Registrar en el logger ────────────────────────────────────────
    registrar_consulta(
        session_id=        session_id,
        pregunta=          pregunta,
        clasificacion=     "PERMITIDA",
        chunks_usados=     chunks,
        respuesta=         respuesta_dict["respuesta"],
        tiempo_ms=         tiempo_ms,
        capa_clasificador= capa_clf
    )

    if not respuesta_dict["exito"]:
        registrar_error(session_id, "llm", respuesta_dict["error"] or "Error desconocido")

    return respuesta_dict["respuesta"]


# ─── Bucle principal de consola ──────────────────────────────────────────────

def main():
    """
    Punto de entrada principal del sistema RAG.

    Muestra el mensaje de bienvenida, verifica que todos los servicios
    estén corriendo, y entra en el bucle de consulta interactivo.
    """
    configurar_salida_utf8()

    print()
    print("=" * 60)
    print("  SISTEMA DE CONSULTA DE REGLAMENTOS INTERNOS")
    print("  RAG Local — 100% Offline")
    print("=" * 60)
    print()

    # ── Verificar servicios antes de iniciar ──────────────────────────────────
    print("  Verificando servicios...\n")

    ollama_ok = _verificar_ollama()
    qdrant_ok = _verificar_qdrant()

    if not ollama_ok or not qdrant_ok:
        print("\n  ❌ No se puede iniciar el sistema.")
        print("     Resuelve los errores anteriores y vuelve a intentarlo.\n")
        sys.exit(1)

    datos_ok = _verificar_datos_indexados()
    if not datos_ok:
        print("\n  ⚠️  El sistema iniciará pero no encontrará respuestas.")
        print("     Recuerda ejecutar: python ingest.py\n")

    print()
    print("  ✅ Sistema listo.")
    print("  Escribe tu pregunta sobre el reglamento.")
    print("  Escribe 'salir' para terminar.\n")
    print("─" * 60)

    session_id = nueva_sesion()

    # ── Bucle principal ───────────────────────────────────────────────────────
    while True:
        try:
            print()
            pregunta = input("  Tu pregunta: ").strip()

            # Salir del sistema
            if pregunta.lower() in ("salir", "exit", "quit", "q"):
                print("\n  👋 Sistema cerrado. ¡Hasta luego!\n")
                break

            # Ignorar entradas vacías
            if not pregunta:
                print("  ⚠️  Por favor escribe una pregunta.")
                continue

            # Procesar la pregunta
            print()
            inicio_respuesta = time.time()
            respuesta = procesar_pregunta(pregunta, session_id)
            tiempo_respuesta = time.time() - inicio_respuesta

            # Mostrar respuesta
            print("─" * 60)
            print()
            print(respuesta)
            print()
            print(f"  ⏱️  Tiempo de respuesta: {tiempo_respuesta:.1f}s")
            print("─" * 60)

        except KeyboardInterrupt:
            print("\n\n  👋 Sistema cerrado. ¡Hasta luego!\n")
            break

        except Exception as e:
            print(f"\n  ❌ Error inesperado: {e}")
            registrar_error(session_id, "main", str(e))
            print("  El sistema sigue activo. Intenta con otra pregunta.\n")


if __name__ == "__main__":
    import sys
    if "--server" in sys.argv:
        import uvicorn
        print("  Levantando servidor API RAG Plastitec...")
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
    else:
        main()
