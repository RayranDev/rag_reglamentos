"""
classifier.py
-------------
Responsabilidad: Detectar si una pregunta es sensible (legal/disciplinaria)
antes de procesarla con el RAG.

Estrategia en dos capas:
  Capa 1 — Keywords: rápido, sin LLM, cubre casos obvios y preguntas
            claramente informativas (bypass directo).
  Capa 2 — LLM fallback: solo si la capa 1 no es concluyente.
"""

import re
import os
import unicodedata
import ollama
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

LLM_MODEL      = os.getenv("LLM_MODEL",            "llama3")
RISK_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.6"))


# ─── Normalización ────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Elimina tildes y convierte a minúsculas para comparación robusta."""
    return unicodedata.normalize("NFD", texto.lower()) \
           .encode("ascii", "ignore").decode("ascii")


# ─── Keywords que indican pregunta CLARAMENTE INFORMATIVA ─────────────────────
# Si la pregunta contiene alguno de estos patrones → PERMITIDA directamente
# sin invocar la Capa 2 (ahorra llamada al LLM).

KEYWORDS_PERMITIDAS = [
    "cuantos dias", "cuantas horas", "como funciona",
    "que dice el rit", "que es", "cual es el horario",
    "cuanto pagan", "que pasa si", "que debo hacer",
    "tengo derecho", "puedo pedir", "como solicito",
    "como pido", "cuanto tiempo", "que dice",
    "cual es la politica", "que establece", "que indica",
    "que contempla", "que norma", "se puede",
    "esta permitido", "esta prohibido", "hay sancion",
]


def _es_claramente_permitida(pregunta: str) -> bool:
    """Retorna True si la pregunta es claramente informativa."""
    p = _normalizar(pregunta)
    return any(kw in p for kw in KEYWORDS_PERMITIDAS)


# ─── Keywords SENSIBLES (casos personales activos) ────────────────────────────

KEYWORDS_SENSIBLES = [
    # Terminación laboral personal activa
    "me van a despedir", "me despidieron", "me quieren despedir",
    "quiero renunciar", "me obligaron a renunciar", "renuncia forzada",
    "me estan presionando para renunciar", "terminaron mi contrato",
    "me quieren sacar", "me van a dar de baja", "me botaron", "me echaron",
    "quiero irme de la empresa",

    # Acciones legales personales
    "quiero poner una tutela", "voy a demandar", "puse una demanda",
    "fui al juzgado", "hable con un abogado", "proceso judicial",
    "quiero ir a la inspeccion del trabajo", "conciliacion laboral",

    # Conflictos interpersonales activos
    "me estan haciendo acoso", "me acosan", "me discriminan",
    "me maltrata", "tengo conflicto con mi jefe", "mi jefe me hostiga",
    "me estan retaliando", "represalia contra mi",
    "me tienen senalado", "me estan haciendo la vida imposible",

    # Casos económicos personales activos
    "no me han pagado", "me deben plata", "me descontaron sin avisar",
    "me retuvieron el salario", "no recibi mis prestaciones",

    # Situaciones disciplinarias personales activas
    "me abrieron un proceso disciplinario", "tengo un descargo",
    "me citaron a descargos", "me pusieron un acta disciplinaria",
    "me van a sancionar a mi", "me pusieron una queja",
    "me levantaron un acta", "me estan mamando gallo",
]

# Compilar patrones normalizados una sola vez
_PATRONES = [
    re.compile(re.escape(_normalizar(kw)), re.IGNORECASE)
    for kw in KEYWORDS_SENSIBLES
]


# ─── Capa 1: Keywords ─────────────────────────────────────────────────────────

def _clasificar_por_keywords(pregunta: str) -> tuple[bool, float]:
    """
    Clasifica la pregunta buscando keywords sensibles normalizadas.

    Returns:
        Tupla (es_sensible: bool, score: float)
    """
    pregunta_norm = _normalizar(pregunta)
    coincidencias = sum(1 for p in _PATRONES if p.search(pregunta_norm))
    score = min(coincidencias / max(1, len(_PATRONES) * RISK_THRESHOLD), 1.0)
    return coincidencias >= 1, score


# ─── Capa 2: LLM fallback ────────────────────────────────────────────────────

def _clasificar_por_llm(pregunta: str) -> bool:
    """
    Usa el LLM para clasificar preguntas ambiguas.
    Solo se invoca cuando la Capa 1 no es concluyente y la pregunta
    no es claramente informativa.
    """
    prompt = f"""Eres un clasificador de preguntas para el sistema de consultas del Reglamento Interno de una empresa colombiana.

Tu única tarea es decidir si una pregunta debe ser BLOQUEADA o PERMITIDA.

PERMITIDA — preguntas informativas sobre normas, políticas o procedimientos:
  - "¿Qué pasa si llego tarde?" → PERMITIDA
  - "¿Cuántos días de vacaciones tengo?" → PERMITIDA
  - "¿Qué debo hacer si me enfermo?" → PERMITIDA
  - "¿Puede haber sanción por faltar?" → PERMITIDA
  - "¿Cómo solicito un permiso?" → PERMITIDA

SENSIBLE — solo cuando hay un caso personal activo con conflicto legal o disciplinario:
  - "Me van a despedir, ¿qué hago?" → SENSIBLE
  - "Quiero poner una tutela contra la empresa" → SENSIBLE
  - "Me abrieron un proceso disciplinario" → SENSIBLE
  - "Me están haciendo acoso laboral" → SENSIBLE

REGLA CLAVE: Si la pregunta usa "¿qué pasa si...?", "¿qué debo hacer...?",
"¿puede haber...?", "¿cómo...?", "¿tengo derecho...?" → es informativa → PERMITIDA.

Pregunta: "{pregunta}"

Responde ÚNICAMENTE con una palabra: PERMITIDA o SENSIBLE"""

    try:
        respuesta = ollama.chat(
            model=   LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options= {"temperature": 0.0, "num_predict": 10}
        )
        texto = respuesta["message"]["content"].strip().upper()
        if "SENSIBLE"  in texto: return True
        if "PERMITIDA" in texto: return False
        return True  # conservador si no responde claramente
    except Exception as e:
        print(f"  ⚠️  Error en clasificador LLM: {e}")
        return False  # en error técnico, dejar pasar


# ─── Función principal ────────────────────────────────────────────────────────

def clasificar_pregunta(pregunta: str) -> dict:
    """
    Clasifica una pregunta usando las dos capas.

    Flujo:
      1. Bypass: si es claramente informativa → PERMITIDA sin LLM
      2. Capa 1: keywords sensibles → SENSIBLE si hay coincidencia
      3. Capa 2: LLM fallback para preguntas ambiguas > 6 palabras

    Returns:
        {"resultado": "PERMITIDA"|"SENSIBLE", "capa_usada": 1|2, "score": float}
    """
    if not pregunta or not pregunta.strip():
        return {"resultado": "PERMITIDA", "capa_usada": 1, "score": 0.0}

    # ── Bypass: claramente informativa ───────────────────────────────────────
    if _es_claramente_permitida(pregunta):
        return {"resultado": "PERMITIDA", "capa_usada": 1, "score": 0.0}

    # ── Capa 1: Keywords sensibles ────────────────────────────────────────────
    es_sensible_kw, score = _clasificar_por_keywords(pregunta)
    if es_sensible_kw:
        return {"resultado": "SENSIBLE", "capa_usada": 1, "score": score}

    # ── Capa 2: LLM fallback ──────────────────────────────────────────────────
    palabras = len(pregunta.split())
    if palabras > 6:
        if _clasificar_por_llm(pregunta):
            return {"resultado": "SENSIBLE", "capa_usada": 2, "score": 0.6}

    return {
        "resultado":  "PERMITIDA",
        "capa_usada": 2 if palabras > 6 else 1,
        "score":      score
    }


# ─── Prueba independiente ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Prueba de classifier.py\n")
    print(f"  Modelo:     {LLM_MODEL}")
    print(f"  Threshold:  {RISK_THRESHOLD}\n")
    print("  " + "─" * 56)

    casos = [
        ("¿Cuál es el horario de entrada?",                   "PERMITIDA"),
        ("¿Cuántos días de vacaciones tengo?",                "PERMITIDA"),
        ("¿Cómo solicito un permiso?",                        "PERMITIDA"),
        ("¿Qué pasa si llego tarde?",                         "PERMITIDA"),
        ("¿Puede haber sanción por faltar?",                  "PERMITIDA"),
        ("¿Qué debo hacer si me siento enfermo?",             "PERMITIDA"),
        ("¿Tengo derecho a licencia de maternidad?",          "PERMITIDA"),
        ("Me van a despedir, ¿qué hago?",                     "SENSIBLE"),
        ("Quiero poner una tutela contra la empresa",         "SENSIBLE"),
        ("Me están haciendo acoso laboral",                   "SENSIBLE"),
        ("Me botaron del trabajo",                            "SENSIBLE"),
        ("Me echaron sin razón",                              "SENSIBLE"),
        ("Me abrieron un proceso disciplinario",              "SENSIBLE"),
        ("¿Cuál es la política de trabajo remoto?",           "PERMITIDA"),
        ("¿A qué hora empieza el turno nocturno?",            "PERMITIDA"),
    ]

    correctos = 0
    for pregunta, esperado in casos:
        r = clasificar_pregunta(pregunta)
        ok = "✅" if r["resultado"] == esperado else "❌"
        if r["resultado"] == esperado: correctos += 1
        print(f"  {ok} [{r['resultado']:<10}] c:{r['capa_usada']} → {pregunta[:52]}")

    print("  " + "─" * 56)
    print(f"\n  Resultado: {correctos}/{len(casos)} correctos\n")