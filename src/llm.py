"""
llm.py
------
Responsabilidad: conectarse a Ollama y generar respuestas
basadas unicamente en el contexto recuperado por el RAG.
"""

import os
import re
import unicodedata
from pathlib import Path

import ollama
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3")


SYSTEM_PROMPT = """Eres el Asistente Virtual de RRHH de PLASTITEC, con acceso al Reglamento Interno de Trabajo (RIT) y documentos corporativos.

IDENTIDAD: Tono profesional, claro y cercano. Espanol colombiano.

REGLAS ABSOLUTAS:
1. Responde unicamente con informacion del contexto proporcionado.
2. Si el contexto no responde la pregunta, di exactamente:
   "No encontre informacion sobre esto en los documentos de Plastitec. Te recomiendo consultarlo con RRHH."
3. Nunca inventes articulos, fechas, politicas ni datos.
4. Puedes explicar reglas del RIT y documentos internos; nunca recomiendes acciones legales ni estrategias juridicas.
5. Nunca respondas sobre salarios individuales o datos personales.
6. No menciones numeros de fragmento, relevancia, ni metadata tecnica.

FORMATO DE RESPUESTA - OBLIGATORIO:
- Empieza directamente con la respuesta. Sin saludos ni preambulos.
- Usa lenguaje natural y humano, no copies el reglamento literalmente.
- Si hay un procedimiento, explicalo en pasos simples.
- Se conciso y no incluyas informacion irrelevante.

Al final incluye siempre:
Fuente: [Articulo o seccion del documento]
Confianza: [alta / media / baja]

Criterio de confianza:
  alta  - el contexto responde directamente la pregunta
  media - el contexto es relacionado pero no exacto
  baja  - el contexto es poco relevante"""


_PATRONES_ARTEFACTOS = [
    r"^[!¡]?excelente pregunta[!.]?\s*",
    r"^hola[!.]?\s*(entiendo que|como asistente|como tu).*?\n",
    r"^entendido[!.]?\s*",
    r"^claro[!.]?\s*",
    r"^por supuesto[!.]?\s*",
    r"^\[tu respuesta en lenguaje natural\]\s*",
    r"^con gusto[!.]?\s*",
    r"fragmento\s+\d+\s*[-—]\s*",
    r"\(relevancia:\s*[\d.]+\)",
    r"relevancia:\s*[\d.]+",
]

_REGEX_ARTEFACTOS = [
    re.compile(patron, re.IGNORECASE | re.MULTILINE)
    for patron in _PATRONES_ARTEFACTOS
]


def _limpiar_respuesta(texto: str) -> str:
    """
    Elimina saludos, preambulos y metadata interna que a veces deja el modelo.
    """
    for regex in _REGEX_ARTEFACTOS:
        texto = regex.sub("", texto)
    return texto.strip()


def _normalizar(texto: str) -> str:
    """Normaliza texto para comparaciones robustas."""
    return unicodedata.normalize("NFD", texto.lower()).encode(
        "ascii", "ignore"
    ).decode("ascii")


def _asegurar_formato_respuesta(texto: str) -> str:
    """
    Garantiza que la respuesta termine con Fuente y Confianza.
    """
    lineas = [linea.strip().lower() for linea in texto.splitlines()]
    tiene_fuente = any(linea.startswith("fuente:") for linea in lineas)
    tiene_confianza = any(linea.startswith("confianza:") for linea in lineas)

    extras = []
    if not tiene_fuente:
        extras.append("Fuente: Fragmentos recuperados")
    if not tiene_confianza:
        extras.append("Confianza: baja")

    if extras:
        return texto.rstrip() + "\n\n" + "\n".join(extras)

    return texto


def _respuesta_regla_directa(contexto: str, pregunta: str) -> str | None:
    """
    Resuelve preguntas muy normativas con reglas deterministas cuando el
    contexto contiene la formula exacta. Esto evita errores de interpretacion
    del LLM en casos criticos y repetitivos del RIT.
    """
    contexto_norm = _normalizar(contexto)
    pregunta_norm = _normalizar(pregunta)

    if (
        "vacaciones" in pregunta_norm
        and ("fecha especifica" in pregunta_norm or "oblig" in pregunta_norm)
        and "epoca de las vacaciones debe ser senalada por plastitec" in contexto_norm
    ):
        return (
            "Si. PLASTITEC puede fijar la fecha o epoca de las vacaciones, "
            "siempre que no perjudique el servicio ni el descanso, y debe "
            "informarla con quince (15) dias de anticipacion.\n\n"
            "Fuente: Artículo 39 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        "embarazo" in pregunta_norm
        and "prueba de gravidez" in contexto_norm
    ):
        return (
            "No. En la admision no se puede exigir prueba de embarazo, salvo "
            "cuando se trate de actividades catalogadas como de alto riesgo.\n\n"
            "Fuente: Artículo 2 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if "libreta militar" in pregunta_norm and "libreta militar" in contexto_norm:
        return (
            "No. La empresa no puede exigir la libreta militar como requisito "
            "de admision o contratacion.\n\n"
            "Fuente: Artículo 2 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        "aprendiz" in pregunta_norm
        and "hayan estado o se encuentren vinculadas laboralmente a la misma" in contexto_norm
    ):
        return (
            "No. La empresa no puede contratar como aprendiz a una persona "
            "que ya haya estado o se encuentre vinculada laboralmente a la misma.\n\n"
            "Fuente: Artículo 15 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        ("extra diurno" in pregunta_norm or "recargo diurno" in pregunta_norm)
        and "veinticinco por ciento (25%)" in contexto_norm
    ):
        return (
            "El trabajo extra diurno se paga con un recargo del veinticinco "
            "por ciento (25%) sobre el valor del trabajo ordinario diurno.\n\n"
            "Fuente: Artículo 32 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        "numero minimo de aprendices" in pregunta_norm
        or ("aprendices" in pregunta_norm and "no contrata" in pregunta_norm)
    ) and "deberan pagar al sena" in contexto_norm:
        return (
            "Si la empresa no cumple la cuota minima obligatoria de aprendices, "
            "debe monetizar esa cuota y pagar al SENA el valor correspondiente "
            "segun la ley aplicable.\n\n"
            "Fuente: Artículo 14 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        "dominical" in pregunta_norm
        and ("90%" in pregunta_norm or "100%" in pregunta_norm)
        and "1o de julio de 2026" in contexto_norm
        and "1o de julio de 2027" in contexto_norm
    ):
        return (
            "El recargo dominical aumentara al 90% desde el 1 de julio de 2026 "
            "y llegara al 100% desde el 1 de julio de 2027.\n\n"
            "Fuente: Artículo 34 del RIT PLASTITEC\n"
            "Confianza: alta"
        )

    if (
        "vestidor" in pregunta_norm or "vestier" in pregunta_norm
    ) and "antes de ingresar al vestier" in contexto_norm:
        return (
            "Para ingresar al vestidor debes entrar con la cofia y la "
            "proteccion auditiva, no ingresar con maquillaje, quitarte la "
            "ropa de calle y las joyas, guardarlas en el locker asignado y "
            "dejar los zapatos de calle dentro de una bolsa plastica.\n\n"
            "Fuente: I-RH-009 Ingreso áreas grises\n"
            "Confianza: alta"
        )

    if (
        "area gris" in pregunta_norm
        and "todo el personal que ingrese al area gris debe cumplir con este instructivo" in contexto_norm
    ):
        return (
            "Para ingresar a un area gris debes cumplir el instructivo de "
            "ingreso: usar solo la dotacion suministrada, colocarte la cofia "
            "y la proteccion auditiva, no entrar con maquillaje, guardar la "
            "ropa de calle y las joyas en el locker, lavar y desinfectar las "
            "manos, ponerte el uniforme, retirar particulas con cinta y "
            "desinfectar los guantes antes de entrar al proceso.\n\n"
            "Fuente: I-RH-009 Ingreso áreas grises\n"
            "Confianza: alta"
        )

    return None


def construir_prompt_usuario(contexto: str, pregunta: str) -> str:
    """
    Arma el mensaje de usuario con contexto y la pregunta original.
    """
    return f"""FRAGMENTOS DEL REGLAMENTO Y DOCUMENTOS DE PLASTITEC:
{contexto}

PREGUNTA:
{pregunta}

INSTRUCCION: Evalua si los fragmentos contienen una regla, prohibicion,
permiso, requisito, excepcion o procedimiento aplicable a la pregunta.
Si la respuesta puede inferirse de forma razonable y directa desde el texto,
respondela sin copiar literalmente. Solo indica que no encontraste la
informacion cuando el contexto realmente no cubra el tema preguntado.
No rellenes con informacion tangencial.
Si la pregunta es de "si/no", "puede/no puede" o "debe/no debe", responde
de forma directa segun la regla del texto. No contradigas el sentido literal
del articulo con interpretaciones conciliadoras.

Es obligatorio terminar con exactamente estas dos lineas:
Fuente: [articulo, seccion o documento usado]
Confianza: [alta / media / baja]"""


def generar_respuesta(contexto: str, pregunta: str) -> dict:
    """
    Genera una respuesta usando el LLM con el contexto del RAG.
    """
    respuesta_directa = _respuesta_regla_directa(contexto, pregunta)
    if respuesta_directa:
        return {
            "respuesta": respuesta_directa,
            "modelo": LLM_MODEL,
            "exito": True,
            "error": None,
        }

    prompt_usuario = construir_prompt_usuario(contexto, pregunta)

    try:
        respuesta_ollama = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario},
            ],
            options={
                "temperature": 0.18,
                "top_p": 0.9,
                "num_predict": 400,
            },
        )

        texto = respuesta_ollama["message"]["content"].strip()
        texto = _limpiar_respuesta(texto)

        if not texto or len(texto) < 10:
            texto = (
                "No encontre informacion sobre esto en los documentos de Plastitec. "
                "Te recomiendo consultarlo con RRHH.\n\n"
                "Fuente: No aplica\nConfianza: baja"
            )

        texto = _asegurar_formato_respuesta(texto)

        return {"respuesta": texto, "modelo": LLM_MODEL, "exito": True, "error": None}

    except Exception as e:
        print(f"  Error al consultar el LLM: {e}")
        return {
            "respuesta": "El servicio de IA no esta disponible en este momento.",
            "modelo": LLM_MODEL,
            "exito": False,
            "error": str(e),
        }


def respuesta_no_encontrada() -> dict:
    """Respuesta estandar cuando no hay chunks relevantes."""
    return {
        "respuesta": (
            "No encontre informacion sobre esto en los documentos de Plastitec.\n\n"
            "Fuente: No aplica\nConfianza: baja"
        ),
        "modelo": LLM_MODEL,
        "exito": True,
        "error": None,
    }


if __name__ == "__main__":
    print("Prueba de llm.py\n")
    print(f"  Modelo: {LLM_MODEL}\n")

    contexto_prueba = """[Fragmento 1 - RIT PLASTITEC 25 NOV 2025.pdf, Pagina 8]
ARTICULO 38. Los trabajadores que hubieren prestado sus servicios durante un (1) ano
tienen derecho a quince (15) dias habiles consecutivos de vacaciones remuneradas.

---

[Fragmento 2 - RIT PLASTITEC 25 NOV 2025.pdf, Pagina 9]
ARTICULO 39. La epoca de las vacaciones debe ser senalada por PLASTITEC a mas tardar
dentro del ano subsiguiente y ellas deben ser concedidas oficiosamente o a peticion
del trabajador, sin perjudicar el servicio."""

    pregunta = "Cuantos dias de vacaciones tengo y como las solicito?"
    print(f"  Pregunta: {pregunta}\n")
    print("  " + "-" * 50)

    resultado = generar_respuesta(contexto_prueba, pregunta)
    print(resultado["respuesta"])
    print("\n  " + "-" * 50)
    print(f"  Exito: {resultado['exito']}")
