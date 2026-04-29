"""
loader.py
---------
Responsabilidad: Leer PDFs desde /data, extraer texto limpio
y guardar los resultados en /processed.

Cambios v2:
  - Normalización de texto antes de guardar (unicode, espacios, guiones)
  - Detección automática de tipo_doc (RIT / BPM / otro) por nombre de archivo
  - Inyecta marcador [TIPO_DOC: X] al inicio del .txt para que el chunker lo lea

Dependencia externa: pymupdf (fitz)
"""

import fitz  # pymupdf
import unicodedata
import re
from pathlib import Path


# ─── Detección de tipo de documento ──────────────────────────────────────────

def _detectar_tipo_doc(nombre_archivo: str) -> str:
    """
    Detecta el tipo de documento por el nombre del archivo PDF.

    Reglas:
      - Contiene "RIT"                              → "RIT"
      - Contiene "BPM", "I-RH", "instructivo",
        "codigo", "I_RH"                            → "BPM"
      - Cualquier otro                              → "otro"

    Args:
        nombre_archivo: Nombre del archivo (con o sin extensión).

    Returns:
        String: "RIT", "BPM" o "otro"
    """
    nombre = nombre_archivo.upper()

    if "RIT" in nombre:
        return "RIT"

    patrones_bpm = ["BPM", "I-RH", "I_RH", "INSTRUCTIVO", "CODIGO",
                    "ETICA", "SST", "CALIDAD"]
    if any(p in nombre for p in patrones_bpm):
        return "BPM"

    return "otro"


# ─── Normalización de texto ───────────────────────────────────────────────────

def _normalizar_texto(texto: str) -> str:
    """
    Limpia el texto extraído del PDF antes de pasarlo al chunker.

    Operaciones:
      1. Normalización unicode NFC (une caracteres compuestos)
      2. Reemplaza guiones tipográficos por guión estándar
      3. Reemplaza comillas tipográficas por comillas estándar
      4. Elimina caracteres de control (excepto \\n y \\t)
      5. Normaliza espacios dentro de cada línea (sin tocar los \\n)
      6. Reduce más de 2 líneas vacías seguidas a máximo 2

    Importante: NO toca los saltos de línea entre líneas con contenido
    porque el chunker los usa para detectar límites de artículo.

    Args:
        texto: Texto crudo extraído del PDF.

    Returns:
        Texto normalizado.
    """
    # 1. Normalización unicode NFC
    texto = unicodedata.normalize("NFC", texto)

    # 2. Guiones tipográficos → guión estándar
    texto = texto.replace("\u2013", "-")   # en dash
    texto = texto.replace("\u2014", "-")   # em dash
    texto = texto.replace("\u2012", "-")   # figure dash

    # 3. Comillas tipográficas → comillas estándar
    texto = texto.replace("\u201c", '"').replace("\u201d", '"')  # " "
    texto = texto.replace("\u2018", "'").replace("\u2019", "'")  # ' '

    # 4. Eliminar caracteres de control excepto \n y \t
    texto = "".join(c for c in texto if c >= " " or c in "\n\t")

    # 5. Normalizar espacios dentro de cada línea (sin tocar \n)
    lineas = texto.split("\n")
    lineas = [" ".join(linea.split()) for linea in lineas]
    texto  = "\n".join(lineas)

    # 6. Reducir líneas vacías excesivas (máximo 2 seguidas)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()


# ─── Extracción de texto ──────────────────────────────────────────────────────

def extraer_texto_pdf(ruta_pdf: Path) -> str:
    """
    Extrae el texto de un PDF página por página con marcadores de página.

    Args:
        ruta_pdf: Ruta al archivo PDF.

    Returns:
        Texto completo con marcadores [PÁGINA X] entre páginas.
        Retorna string vacío si el PDF no tiene texto extraíble.
    """
    texto_completo = []

    try:
        documento = fitz.open(str(ruta_pdf))
    except Exception as e:
        print(f"  ❌ Error al abrir {ruta_pdf.name}: {e}")
        return ""

    for numero_pagina in range(len(documento)):
        pagina = documento[numero_pagina]
        texto  = pagina.get_text()
        if texto.strip():
            texto_completo.append(f"[PÁGINA {numero_pagina + 1}]\n{texto}")

    documento.close()
    return "\n\n".join(texto_completo)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def procesar_pdfs(directorio_data: str, directorio_processed: str) -> list[dict]:
    """
    Procesa todos los PDFs en /data y guarda el texto normalizado en /processed.

    Formato del archivo .txt de salida:
      [TIPO_DOC: RIT]          ← primera línea: marcador de tipo
      [PÁGINA 1]
      ... texto página 1 ...
      [PÁGINA 2]
      ... texto página 2 ...

    Args:
        directorio_data:      Ruta a la carpeta con los PDFs originales.
        directorio_processed: Ruta a la carpeta donde se guarda el texto.

    Returns:
        Lista de dicts por archivo: {nombre, ruta_original, ruta_procesada,
                                     paginas, tipo_doc, estado}
    """
    data_path      = Path(directorio_data)
    processed_path = Path(directorio_processed)

    processed_path.mkdir(parents=True, exist_ok=True)

    archivos_pdf = list(data_path.glob("*.pdf"))

    if not archivos_pdf:
        print("⚠️  No se encontraron archivos PDF en la carpeta /data")
        print(f"   Ruta buscada: {data_path.resolve()}")
        return []

    print(f"📂 Se encontraron {len(archivos_pdf)} PDF(s) en /data\n")

    resultados = []

    for archivo_pdf in archivos_pdf:
        print(f"  📄 Procesando: {archivo_pdf.name}")

        # Detectar tipo de documento antes de extraer
        tipo_doc = _detectar_tipo_doc(archivo_pdf.name)
        print(f"     Tipo detectado: {tipo_doc}")

        texto_crudo = extraer_texto_pdf(archivo_pdf)

        if not texto_crudo.strip():
            print(f"  ⚠️  {archivo_pdf.name} no tiene texto extraíble.")
            print(f"      Puede ser un PDF escaneado. Se omite.\n")
            resultados.append({
                "nombre":         archivo_pdf.name,
                "ruta_original":  str(archivo_pdf),
                "ruta_procesada": None,
                "paginas":        0,
                "tipo_doc":       tipo_doc,
                "estado":         "sin_texto"
            })
            continue

        # Normalizar texto
        texto_normalizado = _normalizar_texto(texto_crudo)

        # Contar páginas
        paginas = texto_normalizado.count("[PÁGINA ")

        # Construir texto final: marcador de tipo + texto normalizado
        texto_final = f"[TIPO_DOC: {tipo_doc}]\n{texto_normalizado}"

        # Guardar en /processed
        nombre_salida = archivo_pdf.stem + ".txt"
        ruta_salida   = processed_path / nombre_salida

        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(texto_final)

        print(f"  ✅ Guardado → /processed/{nombre_salida} "
              f"({paginas} páginas, tipo: {tipo_doc})\n")

        resultados.append({
            "nombre":         archivo_pdf.name,
            "ruta_original":  str(archivo_pdf),
            "ruta_procesada": str(ruta_salida),
            "paginas":        paginas,
            "tipo_doc":       tipo_doc,
            "estado":         "ok"
        })

    return resultados


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    resultados = procesar_pdfs(
        directorio_data=      str(BASE_DIR / "data"),
        directorio_processed= str(BASE_DIR / "processed")
    )

    print("=" * 55)
    print("RESUMEN DE INGESTA")
    print("=" * 55)
    for r in resultados:
        estado = "✅" if r["estado"] == "ok" else "⚠️ "
        print(f"{estado} {r['nombre']:<40} "
              f"{r['paginas']:>3} págs  "
              f"tipo: {r.get('tipo_doc', '?')}")