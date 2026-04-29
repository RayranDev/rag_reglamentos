"""
logger.py
---------
Responsabilidad: Registrar toda la actividad del sistema en archivos
JSON Lines para auditoría empresarial.

Cada consulta queda registrada con: timestamp, pregunta, clasificación,
chunks usados, fuente, y tiempo de respuesta.

Sin dependencias externas — solo stdlib de Python.
"""

import json
import uuid
import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


# ─── Configuración ────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent
LOGS_DIR  = BASE_DIR / "logs"
LOG_FILE  = LOGS_DIR / "consultas.jsonl"

# Máximo 5 MB por archivo, máximo 3 archivos de backup
MAX_BYTES   = 5 * 1024 * 1024
BACKUP_COUNT = 3


# ─── Inicialización ───────────────────────────────────────────────────────────

def inicializar_logger() -> logging.Logger:
    """
    Crea y configura el logger con rotación automática de archivos.

    Crea la carpeta /logs si no existe.
    Usa RotatingFileHandler para evitar que los logs crezcan sin límite.

    Returns:
        Logger configurado listo para usar.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag_reglamentos")
    logger.setLevel(logging.DEBUG)

    # Evitar agregar handlers duplicados si se llama varias veces
    if not logger.handlers:
        handler = RotatingFileHandler(
            filename=     str(LOG_FILE),
            maxBytes=     MAX_BYTES,
            backupCount=  BACKUP_COUNT,
            encoding=     "utf-8"
        )
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    return logger


# Instancia global del logger
_logger = inicializar_logger()


# ─── Funciones públicas ───────────────────────────────────────────────────────

def nueva_sesion() -> str:
    """
    Genera un ID único para identificar una sesión de consulta.

    Returns:
        String UUID4 como identificador de sesión.
    """
    return str(uuid.uuid4())


def registrar_consulta(
    session_id:        str,
    pregunta:          str,
    clasificacion:     str,
    chunks_usados:     list[dict] | None = None,
    respuesta:         str | None        = None,
    tiempo_ms:         float | None      = None,
    capa_clasificador: int | None        = None
) -> None:
    """
    Registra una consulta completa en el archivo de logs.

    Escribe una línea JSON por consulta en logs/consultas.jsonl.
    El formato JSON Lines permite procesar el log línea por línea.

    Args:
        session_id:        ID único de la sesión (generado con nueva_sesion()).
        pregunta:          Texto exacto de la pregunta del usuario.
        clasificacion:     "PERMITIDA" o "SENSIBLE".
        chunks_usados:     Lista de chunks recuperados (puede ser None si fue bloqueada).
        respuesta:         Texto de la respuesta generada (None si fue bloqueada).
        tiempo_ms:         Tiempo total de procesamiento en milisegundos.
        capa_clasificador: Qué capa del clasificador tomó la decisión (1 o 2).
    """
    # Extraer solo los metadatos de los chunks, no el texto completo
    fuentes = []
    if chunks_usados:
        fuentes = [
            {
                "fuente":    c.get("fuente", "desconocido"),
                "pagina":    c.get("pagina", 0),
                "chunk_id":  c.get("chunk_id", ""),
                "score":     round(c.get("score", 0.0), 4)
            }
            for c in chunks_usados
        ]

    entrada = {
        "session_id":        session_id,
        "timestamp":         datetime.now().isoformat(),
        "pregunta":          pregunta,
        "clasificacion":     clasificacion,
        "capa_clasificador": capa_clasificador,
        "fuentes_usadas":    fuentes,
        "respuesta_generada": respuesta is not None,
        "tiempo_ms":         round(tiempo_ms, 2) if tiempo_ms else None
    }

    _logger.info(json.dumps(entrada, ensure_ascii=False))


def registrar_error(session_id: str, modulo: str, error: str) -> None:
    """
    Registra un error del sistema para diagnóstico.

    Args:
        session_id: ID de la sesión donde ocurrió el error.
        modulo:     Nombre del módulo que generó el error (ej: "retriever").
        error:      Mensaje de error.
    """
    entrada = {
        "session_id": session_id,
        "timestamp":  datetime.now().isoformat(),
        "tipo":       "ERROR",
        "modulo":     modulo,
        "detalle":    error
    }
    _logger.error(json.dumps(entrada, ensure_ascii=False))


# ─── Prueba independiente ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Prueba de logger.py\n")

    sid = nueva_sesion()
    print(f"  Session ID generado: {sid}\n")

    # Simular consulta PERMITIDA
    registrar_consulta(
        session_id=        sid,
        pregunta=          "¿Cuál es el horario laboral?",
        clasificacion=     "PERMITIDA",
        chunks_usados=     [
            {"fuente": "reglamento.pdf", "pagina": 3,
             "chunk_id": "reglamento_001", "score": 0.91}
        ],
        respuesta=         "El horario es de 8am a 5pm.",
        tiempo_ms=         340.5,
        capa_clasificador= 1
    )
    print("  ✅ Consulta PERMITIDA registrada")

    # Simular consulta SENSIBLE
    sid2 = nueva_sesion()
    registrar_consulta(
        session_id=        sid2,
        pregunta=          "Me quieren despedir, ¿qué hago?",
        clasificacion=     "SENSIBLE",
        chunks_usados=     None,
        respuesta=         None,
        tiempo_ms=         12.0,
        capa_clasificador= 1
    )
    print("  ✅ Consulta SENSIBLE registrada")

    # Simular error
    registrar_error(sid, "retriever", "Qdrant no responde en localhost:6333")
    print("  ✅ Error registrado\n")

    print(f"  📁 Logs guardados en: {LOG_FILE}")
    print("\n  Contenido del log:")
    print("  " + "─" * 46)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            datos = json.loads(linea)
            print(f"  [{datos.get('clasificacion', datos.get('tipo',''))}] "
                  f"{datos['timestamp'][:19]} — "
                  f"{datos.get('pregunta', datos.get('detalle',''))[:50]}")