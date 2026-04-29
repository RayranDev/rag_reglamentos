"""
evaluar_rit.py
--------------
Script de evaluación masiva del sistema RAG.
Procesa todas las preguntas del CSV, mide tiempos, 
guarda respuestas y chunks utilizados.

Ejecutar: python evaluar_rit.py
Resultado: evaluation/resultados_RAG.csv
"""

import sys
import csv
import json
import time
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from console    import configurar_salida_utf8
from classifier import clasificar_pregunta
from retriever  import buscar_chunks_relevantes, formatear_contexto
from llm        import generar_respuesta, respuesta_no_encontrada
from logger     import registrar_consulta, nueva_sesion

try:
    import psutil
except ImportError:
    psutil = None

# ── Configuración ─────────────────────────────────────────────────────────────

ARCHIVO_PREGUNTAS = "preguntas_RIT_PLASTITEC__1_.csv"
DIRECTORIO_SALIDA = Path("evaluation")
RUN_ID            = datetime.now().strftime('%Y%m%d_%H%M%S')
ARCHIVO_SALIDA    = DIRECTORIO_SALIDA / f"resultados_RAG_{RUN_ID}.csv"
ARCHIVO_RESUMEN   = DIRECTORIO_SALIDA / f"resumen_maquina_{RUN_ID}.json"

COLUMNAS_SALIDA = [
    "id", "categoria", "pregunta", "articulo_referencia",
    "clasificacion", "capa_clasificador",
    "respuesta", "fuente_respuesta",
    "chunks_utilizados", "chunks_detalle_json",
    "tiempo_respuesta_seg", "tiempo_clasificador_seg",
    "tiempo_retriever_seg", "tiempo_llm_seg",
    "cpu_sistema_pct", "ram_sistema_pct",
    "cpu_python_seg", "ram_python_mb",
    "cpu_ollama_seg", "ram_ollama_mb",
    "disco_usado_pct", "disco_libre_gb",
    "disco_read_mb", "disco_write_mb",
    "io_python_read_mb", "io_python_write_mb",
    "gpu_util_pct", "gpu_mem_mb", "gpu_mem_total_mb",
    "nivel_confianza", "notas"
]

MENSAJE_SENSIBLE = (
    "Para este tipo de consultas, por favor dirígete al área de "
    "Recursos Humanos, donde podrán brindarte una orientación más personalizada."
)


# ── Métricas de recursos ──────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent


def _bytes_a_mb(valor: float) -> float:
    return valor / (1024 * 1024)


def _bytes_a_gb(valor: float) -> float:
    return valor / (1024 * 1024 * 1024)


def _tamano_directorio_mb(ruta: Path) -> float:
    if not ruta.exists():
        return 0.0

    total = 0
    for archivo in ruta.rglob("*"):
        try:
            if archivo.is_file():
                total += archivo.stat().st_size
        except OSError:
            continue
    return round(_bytes_a_mb(total), 2)


def _snapshot_gpu() -> dict:
    """
    Lee uso de GPU NVIDIA si nvidia-smi está disponible.

    Si no hay GPU NVIDIA o el driver no expone nvidia-smi, retorna campos vacíos.
    """
    if shutil.which("nvidia-smi") is None:
        return {"gpu_util_pct": "", "gpu_mem_mb": "", "gpu_mem_total_mb": ""}

    try:
        resultado = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:
        return {"gpu_util_pct": "", "gpu_mem_mb": "", "gpu_mem_total_mb": ""}

    util = []
    mem_usada = []
    mem_total = []
    for linea in resultado.stdout.splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) != 3:
            continue
        try:
            util.append(float(partes[0]))
            mem_usada.append(float(partes[1]))
            mem_total.append(float(partes[2]))
        except ValueError:
            continue

    if not util:
        return {"gpu_util_pct": "", "gpu_mem_mb": "", "gpu_mem_total_mb": ""}

    return {
        "gpu_util_pct": round(sum(util) / len(util), 1),
        "gpu_mem_mb": round(sum(mem_usada), 1),
        "gpu_mem_total_mb": round(sum(mem_total), 1),
    }

class MonitorRecursos:
    """Mide consumo aproximado de CPU/RAM/disco/GPU por pregunta."""

    def __init__(self):
        self.disponible = psutil is not None
        self.proceso = psutil.Process(os.getpid()) if self.disponible else None
        if self.disponible:
            psutil.cpu_percent(interval=None)

    def _cpu_proceso(self, proceso) -> float:
        try:
            tiempos = proceso.cpu_times()
            return tiempos.user + tiempos.system
        except Exception:
            return 0.0

    def _rss_mb(self, proceso) -> float:
        try:
            return proceso.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _grupo_ollama(self) -> dict:
        cpu = 0.0
        ram = 0.0
        for proceso in psutil.process_iter(["name"]):
            try:
                nombre = (proceso.info.get("name") or "").lower()
                if "ollama" in nombre:
                    cpu += self._cpu_proceso(proceso)
                    ram += self._rss_mb(proceso)
            except Exception:
                continue
        return {"cpu": cpu, "ram": ram}

    def _io_proceso(self, proceso) -> dict:
        try:
            io = proceso.io_counters()
            return {"read": io.read_bytes, "write": io.write_bytes}
        except Exception:
            return {"read": 0, "write": 0}

    def _io_disco(self) -> dict:
        try:
            io = psutil.disk_io_counters()
            return {"read": io.read_bytes, "write": io.write_bytes}
        except Exception:
            return {"read": 0, "write": 0}

    def snapshot(self) -> dict:
        if not self.disponible:
            return {}

        ollama = self._grupo_ollama()
        disco = psutil.disk_usage(str(BASE_DIR.anchor or BASE_DIR))
        io_python = self._io_proceso(self.proceso)
        io_disco = self._io_disco()
        gpu = _snapshot_gpu()
        return {
            "t": time.perf_counter(),
            "cpu_python": self._cpu_proceso(self.proceso),
            "ram_python": self._rss_mb(self.proceso),
            "cpu_ollama": ollama["cpu"],
            "ram_ollama": ollama["ram"],
            "cpu_sistema": psutil.cpu_percent(interval=None),
            "ram_sistema": psutil.virtual_memory().percent,
            "disco_usado": disco.percent,
            "disco_libre": disco.free,
            "disco_read": io_disco["read"],
            "disco_write": io_disco["write"],
            "io_python_read": io_python["read"],
            "io_python_write": io_python["write"],
            **gpu,
        }

    def medir(self, antes: dict, despues: dict) -> dict:
        if not self.disponible:
            return {
                "cpu_sistema_pct": "",
                "ram_sistema_pct": "",
                "cpu_python_seg": "",
                "ram_python_mb": "",
                "cpu_ollama_seg": "",
                "ram_ollama_mb": "",
                "disco_usado_pct": "",
                "disco_libre_gb": "",
                "disco_read_mb": "",
                "disco_write_mb": "",
                "io_python_read_mb": "",
                "io_python_write_mb": "",
                "gpu_util_pct": "",
                "gpu_mem_mb": "",
                "gpu_mem_total_mb": "",
            }

        return {
            "cpu_sistema_pct": round(despues["cpu_sistema"], 1),
            "ram_sistema_pct": round(despues["ram_sistema"], 1),
            "cpu_python_seg": round(despues["cpu_python"] - antes["cpu_python"], 3),
            "ram_python_mb": round(despues["ram_python"], 1),
            "cpu_ollama_seg": round(despues["cpu_ollama"] - antes["cpu_ollama"], 3),
            "ram_ollama_mb": round(despues["ram_ollama"], 1),
            "disco_usado_pct": round(despues["disco_usado"], 1),
            "disco_libre_gb": round(_bytes_a_gb(despues["disco_libre"]), 2),
            "disco_read_mb": round(_bytes_a_mb(despues["disco_read"] - antes["disco_read"]), 3),
            "disco_write_mb": round(_bytes_a_mb(despues["disco_write"] - antes["disco_write"]), 3),
            "io_python_read_mb": round(_bytes_a_mb(despues["io_python_read"] - antes["io_python_read"]), 3),
            "io_python_write_mb": round(_bytes_a_mb(despues["io_python_write"] - antes["io_python_write"]), 3),
            "gpu_util_pct": despues["gpu_util_pct"],
            "gpu_mem_mb": despues["gpu_mem_mb"],
            "gpu_mem_total_mb": despues["gpu_mem_total_mb"],
        }


# ── Funciones ─────────────────────────────────────────────────────────────────

def extraer_nivel_confianza(respuesta: str) -> str:
    """Extrae el nivel de confianza del texto de respuesta del LLM."""
    respuesta_lower = respuesta.lower()
    if "confianza: alta" in respuesta_lower:
        return "alta"
    elif "confianza: media" in respuesta_lower:
        return "media"
    elif "confianza: baja" in respuesta_lower:
        return "baja"
    return "no_detectada"


def extraer_fuente(respuesta: str) -> str:
    """Extrae la línea de fuente del texto de respuesta del LLM."""
    for linea in respuesta.split("\n"):
        if linea.strip().lower().startswith("fuente:"):
            return linea.strip()
    return ""


def limpiar_respuesta(respuesta: str) -> str:
    """Elimina las líneas de Fuente y Confianza del texto de respuesta."""
    lineas = []
    for linea in respuesta.split("\n"):
        linea_lower = linea.strip().lower()
        if not linea_lower.startswith("fuente:") and not linea_lower.startswith("confianza:"):
            lineas.append(linea)
    return "\n".join(lineas).strip()


def formatear_chunks_csv(chunks: list[dict]) -> str:
    """Formatea los chunks para el CSV en formato legible."""
    if not chunks:
        return ""
    partes = []
    for c in chunks:
        texto_corto = c['texto'][:80].replace('\n', ' ').strip()
        articulo = c.get("articulo") or "sin articulo"
        origen = c.get("origen", "n/d")
        partes.append(
            f"{c['fuente']} p.{c['pagina']} [{c['score']}] "
            f"{articulo} ({origen}): {texto_corto}..."
        )
    return " | ".join(partes)


def formatear_chunks_json(chunks: list[dict]) -> str:
    """Serializa metadatos de chunks para análisis posterior."""
    detalle = []
    for c in chunks:
        detalle.append({
            "chunk_id": c.get("chunk_id", ""),
            "fuente": c.get("fuente", ""),
            "pagina": c.get("pagina", 0),
            "articulo": c.get("articulo", ""),
            "tipo_doc": c.get("tipo_doc", ""),
            "score": c.get("score", 0),
            "origen": c.get("origen", ""),
            "keyword_hits": c.get("keyword_hits", 0),
        })
    return json.dumps(detalle, ensure_ascii=False)


def procesar_pregunta(pregunta: str, session_id: str) -> dict:
    """
    Procesa una pregunta completa por el pipeline RAG.

    Returns:
        Dict con clasificacion, respuesta, chunks, tiempo y confianza.
    """
    inicio = time.time()
    marca = time.perf_counter()

    # Clasificar
    clf = clasificar_pregunta(pregunta)
    tiempo_clasificador = round(time.perf_counter() - marca, 3)
    clasificacion = clf["resultado"]
    capa_clf      = clf["capa_usada"]

    if clasificacion == "SENSIBLE":
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
        return {
            "clasificacion":       "SENSIBLE",
            "capa_clasificador":   capa_clf,
            "respuesta":           MENSAJE_SENSIBLE,
            "fuente_respuesta":     "No aplica",
            "chunks_utilizados":   "",
            "chunks_detalle_json":  "[]",
            "tiempo_respuesta_seg": round(time.time() - inicio, 2),
            "tiempo_clasificador_seg": tiempo_clasificador,
            "tiempo_retriever_seg": 0.0,
            "tiempo_llm_seg":       0.0,
            "nivel_confianza":     "bloqueada",
            "notas":               "Pregunta bloqueada por clasificador"
        }

    # Recuperar chunks
    marca = time.perf_counter()
    chunks = buscar_chunks_relevantes(pregunta)
    tiempo_retriever = round(time.perf_counter() - marca, 3)

    if not chunks:
        respuesta_dict = respuesta_no_encontrada()
        tiempo_seg = round(time.time() - inicio, 2)
        registrar_consulta(
            session_id=        session_id,
            pregunta=          pregunta,
            clasificacion=     "PERMITIDA",
            chunks_usados=     [],
            respuesta=         respuesta_dict["respuesta"],
            tiempo_ms=         tiempo_seg * 1000,
            capa_clasificador= capa_clf
        )
        return {
            "clasificacion":       "PERMITIDA",
            "capa_clasificador":   capa_clf,
            "respuesta":           respuesta_dict["respuesta"],
            "fuente_respuesta":     "Fuente: No aplica",
            "chunks_utilizados":   "",
            "chunks_detalle_json":  "[]",
            "tiempo_respuesta_seg": tiempo_seg,
            "tiempo_clasificador_seg": tiempo_clasificador,
            "tiempo_retriever_seg": tiempo_retriever,
            "tiempo_llm_seg":       0.0,
            "nivel_confianza":     "baja",
            "notas":               "Sin chunks relevantes encontrados"
        }

    # Generar respuesta
    contexto      = formatear_contexto(chunks)
    marca = time.perf_counter()
    respuesta_dict = generar_respuesta(contexto, pregunta)
    tiempo_llm = round(time.perf_counter() - marca, 3)
    tiempo_seg    = round(time.time() - inicio, 2)

    registrar_consulta(
        session_id=        session_id,
        pregunta=          pregunta,
        clasificacion=     "PERMITIDA",
        chunks_usados=     chunks,
        respuesta=         respuesta_dict["respuesta"],
        tiempo_ms=         tiempo_seg * 1000,
        capa_clasificador= capa_clf
    )

    texto_respuesta  = respuesta_dict["respuesta"]
    nivel_confianza  = extraer_nivel_confianza(texto_respuesta)
    fuente           = extraer_fuente(texto_respuesta)
    respuesta_limpia = limpiar_respuesta(texto_respuesta)
    chunks_formateados = formatear_chunks_csv(chunks)
    chunks_detalle_json = formatear_chunks_json(chunks)

    notas = ""
    if not respuesta_dict["exito"]:
        notas = f"Error LLM: {respuesta_dict.get('error', '')}"
    elif nivel_confianza == "baja":
        notas = "Confianza baja — revisar manualmente"

    return {
        "clasificacion":       "PERMITIDA",
        "capa_clasificador":   capa_clf,
        "respuesta":           respuesta_limpia,
        "fuente_respuesta":     fuente,
        "chunks_utilizados":   chunks_formateados,
        "chunks_detalle_json":  chunks_detalle_json,
        "tiempo_respuesta_seg": tiempo_seg,
        "tiempo_clasificador_seg": tiempo_clasificador,
        "tiempo_retriever_seg": tiempo_retriever,
        "tiempo_llm_seg":       tiempo_llm,
        "nivel_confianza":     nivel_confianza,
        "notas":               notas
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configurar_salida_utf8()
    DIRECTORIO_SALIDA.mkdir(exist_ok=True)

    # Leer preguntas
    preguntas = []
    with open(ARCHIVO_PREGUNTAS, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            preguntas.append(fila)

    limite = int(os.getenv("EVAL_LIMIT", "0"))
    if limite > 0:
        preguntas = preguntas[:limite]

    total      = len(preguntas)
    session_id = nueva_sesion()

    print()
    print("=" * 60)
    print("  EVALUACIÓN MASIVA — RAG REGLAMENTOS PLASTITEC")
    print("=" * 60)
    print(f"  Preguntas a procesar: {total}")
    if limite > 0:
        print(f"  Modo prueba:           primeras {limite} preguntas")
    print(f"  Archivo de salida:    {ARCHIVO_SALIDA}")
    print(f"  Session ID:           {session_id}")
    if psutil is None:
        print("  Métricas de recursos: desactivadas (instala psutil)")
    else:
        print("  Métricas de recursos: activadas")
    print()

    resultados    = []
    inicio_total  = time.time()
    tiempos       = []
    monitor       = MonitorRecursos()

    with open(ARCHIVO_SALIDA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_SALIDA, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, fila in enumerate(preguntas, 1):
            pregunta_id  = fila.get("id", str(i))
            categoria    = fila.get("categoria", "")
            pregunta     = fila.get("pregunta", "")
            articulo_ref = fila.get("articulo_referencia", "")

            print(f"  [{i:03d}/{total}] {pregunta[:65]}...")

            recursos_antes = monitor.snapshot()
            resultado = procesar_pregunta(pregunta, session_id)
            recursos_despues = monitor.snapshot()
            metricas = monitor.medir(recursos_antes, recursos_despues)
            tiempos.append(resultado["tiempo_respuesta_seg"])

            fila_salida = {
                "id":                  pregunta_id,
                "categoria":           categoria,
                "pregunta":            pregunta,
                "articulo_referencia": articulo_ref,
                "clasificacion":       resultado["clasificacion"],
                "capa_clasificador":   resultado["capa_clasificador"],
                "respuesta":           resultado["respuesta"],
                "fuente_respuesta":     resultado["fuente_respuesta"],
                "chunks_utilizados":   resultado["chunks_utilizados"],
                "chunks_detalle_json":  resultado["chunks_detalle_json"],
                "tiempo_respuesta_seg": resultado["tiempo_respuesta_seg"],
                "tiempo_clasificador_seg": resultado["tiempo_clasificador_seg"],
                "tiempo_retriever_seg": resultado["tiempo_retriever_seg"],
                "tiempo_llm_seg":       resultado["tiempo_llm_seg"],
                "cpu_sistema_pct":      metricas["cpu_sistema_pct"],
                "ram_sistema_pct":      metricas["ram_sistema_pct"],
                "cpu_python_seg":       metricas["cpu_python_seg"],
                "ram_python_mb":        metricas["ram_python_mb"],
                "cpu_ollama_seg":       metricas["cpu_ollama_seg"],
                "ram_ollama_mb":        metricas["ram_ollama_mb"],
                "disco_usado_pct":      metricas["disco_usado_pct"],
                "disco_libre_gb":       metricas["disco_libre_gb"],
                "disco_read_mb":        metricas["disco_read_mb"],
                "disco_write_mb":       metricas["disco_write_mb"],
                "io_python_read_mb":    metricas["io_python_read_mb"],
                "io_python_write_mb":   metricas["io_python_write_mb"],
                "gpu_util_pct":         metricas["gpu_util_pct"],
                "gpu_mem_mb":           metricas["gpu_mem_mb"],
                "gpu_mem_total_mb":     metricas["gpu_mem_total_mb"],
                "nivel_confianza":     resultado["nivel_confianza"],
                "notas":               resultado["notas"]
            }

            writer.writerow(fila_salida)
            f.flush()  # Guardar progresivamente
            resultados.append(fila_salida)

            estado = "🔴 BLOQ" if resultado["clasificacion"] == "SENSIBLE" else f"✅ {resultado['nivel_confianza']}"
            print(
                f"         {estado} — {resultado['tiempo_respuesta_seg']}s "
                f"(clf {resultado['tiempo_clasificador_seg']}s, "
                f"ret {resultado['tiempo_retriever_seg']}s, "
                f"llm {resultado['tiempo_llm_seg']}s)"
            )

    # Resumen final
    tiempo_total  = round(time.time() - inicio_total, 1)
    tiempo_prom   = round(sum(tiempos) / len(tiempos), 2) if tiempos else 0
    tiempo_max    = round(max(tiempos), 2) if tiempos else 0
    tiempo_min    = round(min(tiempos), 2) if tiempos else 0

    def promedio(campo: str) -> float:
        valores = []
        for fila in resultados:
            valor = fila.get(campo, "")
            if valor == "":
                continue
            try:
                valores.append(float(valor))
            except ValueError:
                continue
        return round(sum(valores) / len(valores), 3) if valores else 0.0

    resumen_maquina = {
        "run_id": RUN_ID,
        "preguntas_procesadas": total,
        "archivo_resultados": str(ARCHIVO_SALIDA),
        "tiempos": {
            "total_seg": tiempo_total,
            "promedio_seg": tiempo_prom,
            "min_seg": tiempo_min,
            "max_seg": tiempo_max,
            "promedio_clasificador_seg": promedio("tiempo_clasificador_seg"),
            "promedio_retriever_seg": promedio("tiempo_retriever_seg"),
            "promedio_llm_seg": promedio("tiempo_llm_seg"),
        },
        "recursos_promedio": {
            "cpu_sistema_pct": promedio("cpu_sistema_pct"),
            "ram_sistema_pct": promedio("ram_sistema_pct"),
            "cpu_python_seg": promedio("cpu_python_seg"),
            "ram_python_mb": promedio("ram_python_mb"),
            "cpu_ollama_seg": promedio("cpu_ollama_seg"),
            "ram_ollama_mb": promedio("ram_ollama_mb"),
            "disco_usado_pct": promedio("disco_usado_pct"),
            "disco_read_mb": promedio("disco_read_mb"),
            "disco_write_mb": promedio("disco_write_mb"),
            "gpu_util_pct": promedio("gpu_util_pct"),
            "gpu_mem_mb": promedio("gpu_mem_mb"),
            "gpu_mem_total_mb": promedio("gpu_mem_total_mb"),
        },
        "almacenamiento_mb": {
            "data": _tamano_directorio_mb(BASE_DIR / "data"),
            "processed": _tamano_directorio_mb(BASE_DIR / "processed"),
            "chunks": _tamano_directorio_mb(BASE_DIR / "chunks"),
            "vectorstore": _tamano_directorio_mb(BASE_DIR / "vectorstore"),
            "logs": _tamano_directorio_mb(BASE_DIR / "logs"),
            "evaluation": _tamano_directorio_mb(BASE_DIR / "evaluation"),
        },
    }

    with open(ARCHIVO_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(resumen_maquina, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("  EVALUACIÓN COMPLETADA")
    print("=" * 60)
    print(f"  Preguntas procesadas:  {total}")
    print(f"  Tiempo total:          {tiempo_total}s")
    print(f"  Tiempo promedio:       {tiempo_prom}s por pregunta")
    print(f"  Tiempo mínimo:         {tiempo_min}s")
    print(f"  Tiempo máximo:         {tiempo_max}s")
    print(f"  Archivo generado:      {ARCHIVO_SALIDA}")
    print(f"  Resumen máquina:       {ARCHIVO_RESUMEN}")
    print()


if __name__ == "__main__":
    main()
