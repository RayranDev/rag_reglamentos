"""
retriever.py
------------
Responsabilidad: convertir la pregunta del usuario en consultas
vectoriales y recuperar los chunks mas relevantes.

Incluye:
- Query expansion con LLM cuando hace falta
- Busqueda hibrida: semantica + keywords curadas
- Filtro automatico por tipo de documento (RIT/BPM)
- Cache de chunks en memoria
"""

import json
import os
import unicodedata
from pathlib import Path

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reglamentos")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
TOP_K = int(os.getenv("TOP_K", "8"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.72"))


def _normalizar(texto: str) -> str:
    """Elimina tildes y convierte a minusculas."""
    return unicodedata.normalize("NFD", texto.lower()).encode(
        "ascii", "ignore"
    ).decode("ascii")


TERMINOS_BPM = [
    "bpm",
    "cofia",
    "maquillaje",
    "area gris",
    "area negra",
    "area blanca",
    "lavado de manos",
    "overol",
    "polainas",
    "locker",
    "vestidor",
    "vestier",
    "manufactura",
    "contaminacion",
]

TERMINOS_RIT = [
    "vacaciones",
    "permiso",
    "licencia",
    "cita medica",
    "medico",
    "llegue tarde",
    "llegar tarde",
    "retardo",
    "sancion",
    "falta",
    "horario",
    "jornada",
    "horas extra",
    "nocturno",
    "salario",
    "domingo",
    "dominical",
    "festivo",
    "sabado",
    "contrato",
    "despido",
    "descargo",
    "disciplinario",
    "teletrabajo",
    "trabajo remoto",
    "trabajo en casa",
    "acoso",
    "reclamo",
    # Admision y seleccion
    "prueba",
    "embarazo",
    "gravidez",
    "admision",
    "aspirante",
    "documentos",
    "seleccion",
    "psicotecnica",
    "psicologica",
    "libreta militar",
    "requisito",
    "hoja de vida",
    # Derechos y prestaciones
    "prestaciones",
    "cesantias",
    "prima",
    "seguridad social",
    "incapacidad",
    "dotacion",
    "auxilio",
    "subsidio",
    # Evaluacion y desempeno
    "evaluacion",
    "desempeno",
    "periodo de prueba",
    # Fueros
    "fuero",
    "maternidad",
    "paternidad",
    "sindicato",
    "embarazada",
    # General RIT
    "reglamento",
    "rit",
    "norma",
    "politica",
    "obligacion",
    "prohibicion",
    "derecho del trabajador",
    "puede la empresa",
]


def _inferir_tipo_doc(pregunta: str) -> str | None:
    """
    Infiere un filtro de documento por dominio para reducir ruido.
    """
    pregunta_norm = _normalizar(pregunta)

    if any(termino in pregunta_norm for termino in TERMINOS_BPM):
        return "BPM"
    if any(termino in pregunta_norm for termino in TERMINOS_RIT):
        return "RIT"

    return None


_cache_chunks: list[dict] = []


def _cargar_chunks_cache() -> list[dict]:
    """
    Carga los chunks desde disco la primera vez y luego los sirve desde memoria.
    """
    global _cache_chunks
    if _cache_chunks:
        return _cache_chunks

    chunks_path = BASE_DIR / "chunks"
    for archivo in chunks_path.glob("*.json"):
        with open(archivo, "r", encoding="utf-8") as f:
            _cache_chunks.extend(json.load(f))

    return _cache_chunks


def expandir_query(pregunta: str) -> list[str]:
    """
    Reformula la pregunta en variantes tecnicas del RIT colombiano.
    Retorna [pregunta_original] + hasta 3 variantes.
    """
    prompt = f"""Eres un experto en reglamentos laborales colombianos.
Un empleado hizo esta pregunta: "{pregunta}"

Genera exactamente 3 reformulaciones usando terminologia tecnica y formal
de reglamentos internos de trabajo colombianos.
Cada reformulacion debe capturar el mismo concepto con palabras distintas.

Responde unicamente con las 3 reformulaciones, una por linea,
sin numeracion ni explicaciones."""

    try:
        respuesta = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 120},
        )
        texto = respuesta["message"]["content"].strip()
        variantes = [v.strip() for v in texto.split("\n") if v.strip()][:3]
        return [pregunta] + variantes
    except Exception:
        return [pregunta]


def _generar_embedding(texto: str) -> list[float] | None:
    """Genera embedding con Ollama."""
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]
    except Exception as e:
        print(f"  Error generando embedding: {e}")
        return None


def _buscar_en_qdrant(
    vector: list[float],
    top_k: int,
    fuente_filtro: str | None = None,
    tipo_doc_filtro: str | None = None,
) -> list:
    """Ejecuta busqueda vectorial en Qdrant con filtros opcionales."""
    try:
        cliente = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"  No se pudo conectar a Qdrant: {e}")
        return []

    must = []
    if fuente_filtro:
        must.append(FieldCondition(key="fuente", match=MatchValue(value=fuente_filtro)))
    if tipo_doc_filtro:
        must.append(
            FieldCondition(key="tipo_doc", match=MatchValue(value=tipo_doc_filtro))
        )
    filtro = Filter(must=must) if must else None

    try:
        return cliente.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            query_filter=filtro,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        )
    except Exception as e:
        print(f"  Error en busqueda Qdrant: {e}")
        return []


MAPA_KEYWORDS = {
    # Contratos y admision
    "admision": [
        "ARTICULO 2",
        "aspirante",
        "documentos",
        "pruebas psicotecnicas",
        "pruebas psicologicas",
    ],
    "hoja de vida": [
        "ARTICULO 2",
        "aspirante",
        "hoja de vida",
        "cedula",
        "certificado del ultimo empleador",
    ],
    "cedula": [
        "ARTICULO 2",
        "aspirante",
        "hoja de vida",
        "cedula",
        "certificado del ultimo empleador",
    ],
    "certificado ultimo empleador": [
        "ARTICULO 2",
        "certificado del ultimo empleador",
        "tiempo de servicio",
        "indole de la labor ejecutada",
    ],
    "embarazo": [
        "ARTICULO 2",
        "prueba de gravidez",
        "gravidez",
        "embarazo",
        "alto riesgo",
    ],
    "prueba embarazo": [
        "ARTICULO 2",
        "prueba de gravidez",
        "gravidez",
        "embarazo",
        "alto riesgo",
    ],
    "prueba de embarazo": [
        "ARTICULO 2",
        "prueba de gravidez",
        "gravidez",
        "embarazo",
        "alto riesgo",
    ],
    "libreta militar": ["ARTICULO 2", "libreta militar", "decreto 2150"],
    "psicotecnicas": [
        "ARTICULO 2",
        "pruebas psicotecnicas",
        "psicotecnicas",
        "psicologicas",
    ],
    "psicologicas": [
        "ARTICULO 2",
        "pruebas psicotecnicas",
        "psicotecnicas",
        "psicologicas",
    ],
    "seleccion": [
        "ARTICULO 2",
        "pruebas psicotecnicas",
        "psicotecnicas",
        "psicologicas",
        "documentacion presentada",
    ],
    "contrato": [
        "ARTICULO 3",
        "ARTICULO 4",
        "ARTICULO 5",
        "termino fijo",
        "termino indefinido",
        "obra o labor",
    ],
    "aprendiz": [
        "ARTICULO 10",
        "ARTICULO 13",
        "ARTICULO 15",
        "contrato de aprendizaje",
        "sena",
        "apoyo de sostenimiento",
        "seleccion de aprendices",
    ],
    "aprendices": [
        "ARTICULO 10",
        "ARTICULO 13",
        "ARTICULO 15",
        "contrato de aprendizaje",
        "sena",
        "cuota de aprendices",
        "seleccion de aprendices",
    ],
    "cuota aprendices": [
        "ARTICULO 13",
        "cuota de aprendices",
        "numero minimo obligatorio",
        "monetizacion",
    ],
    "numero minimo de aprendices": [
        "ARTICULO 13",
        "ARTICULO 14",
        "numero minimo obligatorio de aprendices",
        "monetizacion de la cuota de aprendizaje",
        "pagar al sena",
    ],
    "no contrata aprendices": [
        "ARTICULO 14",
        "monetizacion de la cuota de aprendizaje",
        "deberan pagar al sena",
    ],
    "no contrata": [
        "ARTICULO 14",
        "monetizacion de la cuota de aprendizaje",
        "deberan pagar al sena",
    ],
    "monetizacion": [
        "ARTICULO 14",
        "monetizacion de la cuota de aprendizaje",
        "deberan pagar al sena",
    ],
    "ya trabajo": [
        "ARTICULO 15",
        "seleccion de aprendices",
        "hayan estado",
        "vinculadas laboralmente a la misma",
    ],
    "ya trabajo en la empresa": [
        "ARTICULO 15",
        "seleccion de aprendices",
        "hayan estado",
        "vinculadas laboralmente a la misma",
    ],
    "trabajo en la empresa": [
        "ARTICULO 15",
        "hayan estado",
        "vinculadas laboralmente a la misma",
    ],
    "prueba": ["periodo de prueba", "ARTICULO 6", "ARTICULO 8"],
    "periodo prueba": ["periodo de prueba", "ARTICULO 8", "dos meses"],
    # Jornada
    "tarde": ["retardo", "ARTICULO 88", "llamado de atencion", "hora de entrada"],
    "retardo": [
        "retardo",
        "ARTICULO 88",
        "ARTICULO 89",
        "llamado de atencion",
        "hora de entrada",
    ],
    "hora entrada": [
        "retardo",
        "ARTICULO 88",
        "llamado de atencion",
        "hora de entrada",
    ],
    "sancion": [
        "sancion",
        "suspension",
        "ARTICULO 88",
        "llamado de atencion",
        "falta leve",
    ],
    "jornada": ["jornada laboral", "ARTICULO 22", "horas semanales"],
    "horario": ["ARTICULO 22", "jornada laboral", "horario"],
    "turno": [
        "ARTICULO 22",
        "ARTICULO 24",
        "turnos de trabajo",
        "modificar los actuales",
        "cambiar los horarios",
    ],
    "horas extra": [
        "horas extras",
        "trabajo suplementario",
        "ARTICULO 29",
        "ARTICULO 30",
        "ARTICULO 31",
        "ARTICULO 33",
    ],
    "horas extras": [
        "horas extras",
        "trabajo suplementario",
        "ARTICULO 29",
        "ARTICULO 30",
        "ARTICULO 31",
        "ARTICULO 33",
    ],
    "sin autorizacion": [
        "ARTICULO 33",
        "no reconocera trabajo suplementario",
        "expresamente lo autorice",
        "jefe inmediato",
        "sin autorizacion",
    ],
    "nocturno": ["trabajo nocturno", "ARTICULO 28", "recargo"],
    "recargo": [
        "ARTICULO 28",
        "ARTICULO 31",
        "trabajo extra diurno",
        "veinticinco por ciento (25%)",
        "trabajo nocturno",
    ],
    "extra diurno": [
        "ARTICULO 31",
        "trabajo extra diurno",
        "veinticinco por ciento (25%)",
        "recargo",
    ],
    "recargo diurno": [
        "ARTICULO 31",
        "trabajo extra diurno",
        "veinticinco por ciento (25%)",
        "recargo",
    ],
    # Ausencias
    "falta": ["inasistencia", "ausencia injustificada", "ARTICULO 88", "falta total"],
    "sin excusa": [
        "sin excusa suficiente",
        "ARTICULO 88",
        "ARTICULO 89",
        "falta total",
    ],
    "enfermo": ["enfermo", "incapacidad", "ARTICULO 59"],
    # Permisos y licencias
    "cita medica": [
        "asistencia a citas medicas",
        "constancia de agendamiento",
        "copago",
        "especialistas",
        "ARTICULO 46",
    ],
    "citas medicas": [
        "asistencia a citas medicas",
        "constancia de agendamiento",
        "copago",
        "especialistas",
        "ARTICULO 46",
    ],
    "cita": [
        "asistencia a citas medicas",
        "constancia de agendamiento",
        "comprobante de asistencia",
    ],
    "servicio medico": [
        "servicio medico correspondiente",
        "ARTICULO 45",
        "permisos",
        "anticipacion",
    ],
    "permiso medico": [
        "asistencia a citas medicas",
        "constancia de agendamiento",
        "copago",
        "especialistas",
    ],
    "permiso": [
        "ARTICULO 45",
        "concedera a sus trabajadores los permisos",
        "sufragio",
        "calamidad domestica",
    ],
    "licencia": ["licencia remunerada", "ARTICULO 46"],
    "maternidad": ["licencia de maternidad", "ARTICULO 46", "semanas"],
    "paternidad": ["licencia de paternidad", "ARTICULO 46"],
    "luto": ["licencia por luto", "ARTICULO 46", "dias habiles", "fallecimiento"],
    "lactancia": ["periodo de lactancia", "ARTICULO 46", "treinta minutos"],
    # Descansos, dominicales y festivos
    "domingo": [
        "ARTICULO 34",
        "ARTICULO 35",
        "ARTICULO 36",
        "descanso dominical",
        "trabajo dominical",
    ],
    "dominical": [
        "ARTICULO 34",
        "ARTICULO 35",
        "ARTICULO 36",
        "perdida del dominical",
        "ARTICULO 26",
    ],
    "90%": [
        "ARTICULO 34",
        "1o de julio de 2026",
        "90%",
        "100%",
    ],
    "100%": [
        "ARTICULO 34",
        "1o de julio de 2027",
        "90%",
        "100%",
    ],
    "recargo dominical": [
        "ARTICULO 34",
        "1o de julio de 2026",
        "1o de julio de 2027",
        "90%",
        "100%",
    ],
    "descanso dominical": [
        "ARTICULO 36",
        "duracion minima de 24 horas",
        "minima de 24 horas",
        "descanso dominical",
    ],
    "tiempo minimo": [
        "ARTICULO 36",
        "duracion minima de 24 horas",
        "minima de 24 horas",
    ],
    "duracion": [
        "ARTICULO 36",
        "duracion minima de 24 horas",
        "minima de 24 horas",
    ],
    "festivo": ["ARTICULO 34", "dias de fiesta", "recargo"],
    "sabado": [
        "ARTICULO 23",
        "ARTICULO 34",
        "sabado o domingo",
        "descanso obligatorio",
    ],
    "perder el pago": [
        "ARTICULO 26",
        "perdida del dominical",
        "ausencia, retardo o interrupcion injustificada",
    ],
    # Vacaciones
    "vacaciones": [
        "vacaciones",
        "dias habiles",
        "ARTICULO 38",
        "ARTICULO 39",
        "epoca de las vacaciones",
    ],
    "fecha especifica": [
        "ARTICULO 39",
        "epoca de las vacaciones",
        "debe ser senalada por PLASTITEC",
        "ano subsiguiente",
    ],
    "cuando me las dan": [
        "ARTICULO 39",
        "epoca de las vacaciones",
        "debe ser senalada por PLASTITEC",
    ],
    # Salario y dotacion
    "salario": ["salario", "ARTICULO 48", "remuneracion"],
    "dotacion": ["dotacion", "ARTICULO 56"],
    "pago": ["pago", "ARTICULO 52", "mensual"],
    # Disciplinario
    "descargo": [
        "ARTICULO 85",
        "diligencia de descargos",
        "citacion escrita",
        "debido proceso",
    ],
    "disciplinario": [
        "proceso disciplinario",
        "ARTICULO 82",
        "ARTICULO 85",
        "ARTICULO 86",
    ],
    "suspension": ["suspension", "ARTICULO 88", "ARTICULO 89"],
    # Fueros y contratos
    "fuero": [
        "estabilidad laboral reforzada",
        "ARTICULO 130",
        "fuero de maternidad",
        "fuero sindical",
    ],
    "sindicato": ["ARTICULO 129", "asociacion", "fuero sindical"],
    "despido": ["terminacion", "justa causa", "ARTICULO 97"],
    "prepensionado": [
        "ARTICULO 130",
        "estabilidad laboral reforzada",
        "pension",
        "prepensionado",
    ],
    # Teletrabajo
    "teletrabajo": ["teletrabajo", "ARTICULO 101", "ARTICULO 102"],
    "trabajo remoto": ["trabajo remoto", "ARTICULO 107"],
    "trabajo en casa": ["trabajo en casa", "ARTICULO 99"],
    "direccion y manejo": [
        "direccion, confianza y manejo",
        "horas extras",
        "ARTICULO 106",
    ],
    # BPM
    "area gris": ["ingreso areas grises", "I-RH-009", "cofia", "locker"],
    "ingreso area gris": ["ingreso areas grises", "I-RH-009", "cofia", "locker"],
    "protocolo area gris": ["ingreso areas grises", "I-RH-009", "cofia", "locker"],
    "area negra": ["ingreso areas negras", "I-RH-010", "overol"],
    "area blanca": ["ingreso areas blancas", "I-RH-011", "polainas"],
    "lavado de manos": ["I-RH-012", "lavado de manos", "jabon antibacterial"],
    "bpm": ["buenas practicas de manufactura", "I-RH-003", "contaminacion"],
    "cofia": ["cofia", "I-RH-009", "I-RH-003"],
    "vestidor": ["vestidor", "locker", "casillero", "I-RH-009", "I-RH-010"],
    "vestier": ["vestidor", "locker", "casillero", "I-RH-009", "I-RH-010"],
    "maquillaje": ["maquillaje", "prohibido", "I-RH-003"],
    "joyas": ["joyas", "prohibido", "I-RH-003"],
    "contaminacion": ["contaminacion", "areas", "I-RH-003"],
    # Acoso
    "acoso": ["acoso laboral", "ARTICULO 112", "Ley 1010"],
    "no acoso": [
        "ARTICULO 115",
        "no constituyen acoso laboral",
        "No constituyen acoso laboral",
    ],
    "no constituye acoso": [
        "ARTICULO 115",
        "no constituyen acoso laboral",
        "No constituyen acoso laboral",
    ],
    # Obligaciones y prohibiciones
    "asuntos personales": [
        "no atender durante las horas de trabajo",
        "asuntos y ocupaciones distintas",
        "horas de trabajo asuntos",
        "previa autorizacion",
    ],
    "retirar documentos": [
        "ARTICULO 79",
        "se prohibe a los trabajadores",
        "documentos de la empresa",
    ],
    "documentos de la empresa": [
        "ARTICULO 79",
        "se prohibe a los trabajadores",
        "documentos de la empresa",
    ],
    "visitas": ["visitas de caracter personal", "personas ajenas", "ARTICULO 60"],
    "personas externas": ["personas ajenas", "instalaciones", "ARTICULO 60"],
    "alcohol": ["bebidas embriagantes", "efectos del alcohol", "ARTICULO 60"],
    "tiendas": ["comprar mercancias", "tiendas", "almacenes", "ARTICULO 58"],
    "almacenes": ["comprar mercancias", "tiendas", "almacenes", "ARTICULO 58"],
    # Reclamos
    "reclamo": ["ARTICULO 110", "reclamo", "Recursos Humanos"],
}


def _keywords_para_pregunta(pregunta: str) -> set[str]:
    """Retorna las keywords curadas que aplican a la pregunta."""
    pregunta_norm = _normalizar(pregunta)
    keywords_buscar = set()

    terminos_especificos_admision = [
        "prueba embarazo",
        "embarazo",
        "psicotecnicas",
        "psicologicas",
        "seleccion",
        "libreta militar",
    ]
    omitir_prueba_generica = any(
        termino in pregunta_norm for termino in terminos_especificos_admision
    )
    terminos_especificos_aprendices = [
        "numero minimo de aprendices",
        "cuota aprendices",
        "no contrata",
        "monetizacion",
    ]
    priorizar_cuota_aprendices = any(
        termino in pregunta_norm for termino in terminos_especificos_aprendices
    )

    for termino, keywords in MAPA_KEYWORDS.items():
        termino_norm = _normalizar(termino)
        if termino_norm == "prueba" and omitir_prueba_generica:
            continue
        if termino_norm in {"aprendiz", "aprendices"} and priorizar_cuota_aprendices:
            continue
        if termino_norm in pregunta_norm:
            keywords_buscar.update(keywords)

    return keywords_buscar


def _buscar_por_keywords(
    pregunta: str,
    tipo_doc_filtro: str | None = None,
    limite: int = 4,
) -> list[dict]:
    """
    Busca coincidencias por keywords curadas.

    Se ejecuta aunque la busqueda semantica ya tenga resultados,
    porque algunos temas puntuales se recuperan mejor por articulo.
    """
    todos_chunks = _cargar_chunks_cache()
    keywords_buscar = _keywords_para_pregunta(pregunta)

    if not keywords_buscar:
        return []

    resultados = []
    for chunk in todos_chunks:
        if tipo_doc_filtro and chunk.get("tipo_doc") != tipo_doc_filtro:
            continue

        texto_norm = _normalizar(chunk["texto"])
        coincidencias = sum(
            1 for kw in keywords_buscar if _normalizar(kw) in texto_norm
        )
        if coincidencias <= 0:
            continue

        score_kw = min(0.78 + (coincidencias * 0.05), 0.94)
        resultados.append(
            {
                **chunk,
                "score": round(score_kw, 4),
                "keyword_hits": coincidencias,
                "origen": "keywords",
            }
        )

    resultados.sort(key=lambda x: (x["keyword_hits"], x["score"]), reverse=True)
    return resultados[:limite]


def buscar_chunks_relevantes(
    pregunta: str,
    top_k: int = TOP_K,
    fuente_filtro: str | None = None,
    tipo_doc_filtro: str | None = None,
) -> list[dict]:
    """
    Busqueda hibrida: semantica + keywords.

    Si la pregunta ya cae en keywords curadas, evitamos query expansion
    para ahorrar tiempo y reducir ruido.
    """
    keywords_buscar = _keywords_para_pregunta(pregunta)
    queries = [pregunta] if keywords_buscar else expandir_query(pregunta)
    vistos: dict[str, dict] = {}
    tipo_doc_resuelto = tipo_doc_filtro or _inferir_tipo_doc(pregunta)

    for query in queries:
        vector = _generar_embedding(query)
        if vector is None:
            continue

        puntos = _buscar_en_qdrant(
            vector,
            top_k * 2,
            fuente_filtro,
            tipo_doc_resuelto,
        )
        for punto in puntos:
            payload = punto.payload or {}
            chunk_id = payload.get("chunk_id", "")
            if chunk_id not in vistos or vistos[chunk_id]["score"] < punto.score:
                vistos[chunk_id] = {
                    "texto": payload.get("texto", ""),
                    "fuente": payload.get("fuente", "desconocido"),
                    "pagina": payload.get("pagina", 0),
                    "chunk_id": chunk_id,
                    "articulo": payload.get("articulo", ""),
                    "tipo_doc": payload.get("tipo_doc", ""),
                    "score": round(punto.score, 4),
                    "origen": "semantico",
                }

    resultados_kw = _buscar_por_keywords(
        pregunta,
        tipo_doc_filtro=tipo_doc_resuelto,
        limite=max(4, top_k // 2),
    )
    for chunk in resultados_kw:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id in vistos:
            hits = chunk.get("keyword_hits", 0)
            vistos[chunk_id]["keyword_hits"] = max(
                vistos[chunk_id].get("keyword_hits", 0),
                hits,
            )
            vistos[chunk_id]["score"] = round(
                max(
                    vistos[chunk_id]["score"],
                    min(chunk.get("score", 0.74) + (hits * 0.01), 0.95),
                ),
                4,
            )
            vistos[chunk_id]["origen"] = "semantico+keywords"
        else:
            vistos[chunk_id] = {
                "texto": chunk.get("texto", ""),
                "fuente": chunk.get("fuente", "desconocido"),
                "pagina": chunk.get("pagina", 0),
                "chunk_id": chunk_id,
                "articulo": chunk.get("articulo", ""),
                "tipo_doc": chunk.get("tipo_doc", ""),
                "score": chunk.get("score", 0.74),
                "keyword_hits": chunk.get("keyword_hits", 0),
                "origen": "keywords",
            }

    resultados = sorted(
        vistos.values(),
        key=lambda x: (x["score"], x.get("keyword_hits", 0)),
        reverse=True,
    )
    return resultados[:top_k]


def formatear_contexto(chunks: list[dict]) -> str:
    """Convierte chunks en texto estructurado para el LLM."""
    if not chunks:
        return "No se encontro informacion relevante."

    bloques = []
    for i, chunk in enumerate(chunks, 1):
        referencia = f"{chunk['fuente']}, Pagina {chunk['pagina']}"
        if chunk.get("articulo"):
            referencia += f", {chunk['articulo']}"
        bloques.append(f"[Fragmento {i} - {referencia}]\n{chunk['texto']}")
    return "\n\n---\n\n".join(bloques)


if __name__ == "__main__":
    print("Prueba de retriever.py\n")

    preguntas = [
        "Cuantos dias de vacaciones tengo?",
        "llegue tarde varias veces, me pueden sancionar?",
        "tengo cita con el medico manana, que debo hacer?",
        "Como ingreso al area gris?",
    ]

    for pregunta in preguntas:
        print(f"\n  {pregunta}")
        chunks = buscar_chunks_relevantes(pregunta)
        if not chunks:
            print("  Sin resultados")
            continue
        for chunk in chunks[:3]:
            print(
                f"     [{chunk['score']}] p.{chunk['pagina']} - "
                f"{chunk['texto'][:75].strip()}..."
            )
