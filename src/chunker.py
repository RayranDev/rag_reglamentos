"""
chunker.py
----------
Responsabilidad: Dividir el texto extraído en chunks con overlap,
garantizando que ningún chunk supere el límite del modelo de embeddings.

Cambios v2:
  - Tamaño aumentado a 1100 caracteres (era 800)
  - Overlap aumentado a 250 (era 160)
  - Prioridad de corte: ARTÍCULO > CAPÍTULO > PARÁGRAFO > \\n\\n > punto
  - Lee el marcador [TIPO_DOC: X] del .txt generado por loader.py
  - Agrega campos `articulo` y `tipo_doc` al metadata de cada chunk

Entrada:  /processed/*.txt
Salida:   /chunks/*.json
"""

import json
import re
from pathlib import Path


# mxbai-embed-large soporta ~512 tokens ≈ 1800 caracteres en español.
# Usamos 1100 para tener margen seguro y que los artículos del RIT
# (promedio 1000-1400 chars) entren completos con contexto de overlap.
LIMITE_CARACTERES  = 1400
OVERLAP_CARACTERES = 250

# Patrones de límites de artículo ordenados por prioridad
_PATRON_ARTICULO  = re.compile(r'\n(ART[IÍ]CULO\s+\d+[A-Z]?\.?\b)', re.IGNORECASE)
_PATRON_CAPITULO  = re.compile(r'\n(CAP[IÍ]TULO\s+[IVXLCDM]+\b)', re.IGNORECASE)
_PATRON_PARAGRAFO = re.compile(r'\n(PAR[AÁ]GRAFO\s*\d*\.?\s)', re.IGNORECASE)

# Patrón para extraer el nombre/número del artículo más cercano al inicio del chunk
_PATRON_NOMBRE_ART = re.compile(
    r'ART[IÍ]CULO\s+(\d+[A-Z]?\.?)\s*[-–.:]?\s*([A-ZÁÉÍÓÚÑ][^\n.]{0,80})?',
    re.IGNORECASE
)


# ─── Lectura de tipo_doc ──────────────────────────────────────────────────────

def _extraer_tipo_doc(texto: str) -> tuple[str, str]:
    """
    Lee el marcador [TIPO_DOC: X] de la primera línea del texto
    y retorna el tipo + el texto sin esa línea.

    Args:
        texto: Texto completo del .txt (primera línea puede ser [TIPO_DOC: X]).

    Returns:
        Tupla (tipo_doc, texto_sin_marcador)
    """
    primera_linea = texto.split("\n")[0].strip()
    match = re.match(r'\[TIPO_DOC:\s*([A-Za-z]+)\]', primera_linea)

    if match:
        tipo_doc       = match.group(1).upper()
        texto_limpio   = texto[len(primera_linea):].lstrip("\n")
        return tipo_doc, texto_limpio

    return "otro", texto


# ─── Lectura de páginas ───────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> tuple[str, dict[int, int]]:
    """
    Elimina los marcadores [PÁGINA X] del texto y construye
    un mapa de posición de carácter → número de página.

    Args:
        texto: Texto con marcadores de página.

    Returns:
        Tupla (texto_limpio, mapa_paginas)
    """
    mapa_paginas = {}
    pagina_actual = 1
    texto_limpio  = []
    pos_actual    = 0

    for linea in texto.split("\n"):
        match = re.match(r'\[PÁGINA (\d+)\]', linea.strip())
        if match:
            pagina_actual = int(match.group(1))
        else:
            mapa_paginas[pos_actual] = pagina_actual
            texto_limpio.append(linea)
            pos_actual += len(linea) + 1

    return "\n".join(texto_limpio), mapa_paginas


def obtener_pagina_en_posicion(pos: int, mapa_paginas: dict) -> int:
    """
    Retorna el número de página para una posición de carácter.

    Args:
        pos:          Posición del carácter en el texto limpio.
        mapa_paginas: Mapa construido por limpiar_texto().

    Returns:
        Número de página.
    """
    pagina = 1
    for posicion_marcador, numero_pagina in sorted(mapa_paginas.items()):
        if posicion_marcador <= pos:
            pagina = numero_pagina
        else:
            break
    return pagina


# ─── Extracción de artículo ───────────────────────────────────────────────────

def _extraer_articulo(fragmento: str) -> str:
    """
    Extrae el identificador del artículo más cercano al inicio del chunk.

    Busca el primer ARTÍCULO N dentro del fragmento y retorna
    "Artículo N — Título" si tiene título, o "Artículo N" si no.

    Args:
        fragmento: Texto del chunk.

    Returns:
        String con la referencia del artículo, o "" si no hay ninguno.
    """
    match = _PATRON_NOMBRE_ART.search(fragmento[:400])  # Solo buscar al inicio
    if not match:
        return ""

    numero = match.group(1).rstrip(".")
    titulo = match.group(2)

    if titulo:
        titulo = titulo.strip().rstrip(".")
        # Limitar título a 60 chars para que no sea demasiado largo
        if len(titulo) > 60:
            titulo = titulo[:57] + "..."
        return f"Artículo {numero} — {titulo}"

    return f"Artículo {numero}"


def _segmentar_por_articulos(texto: str) -> list[tuple[int, int, str]]:
    """
    Segmenta el texto usando el inicio de cada artículo como frontera principal.

    El bloque previo al primer artículo se conserva como preámbulo sin artículo.
    Esto permite que, si luego un artículo debe partirse por longitud, todos sus
    subchunks hereden la misma referencia normativa.
    """
    coincidencias = list(_PATRON_ARTICULO.finditer(texto))
    if not coincidencias:
        return [(0, len(texto), "")]

    bloques: list[tuple[int, int, str]] = []

    if coincidencias[0].start() > 0:
        bloques.append((0, coincidencias[0].start(), ""))

    for i, match in enumerate(coincidencias):
        inicio = match.start()
        fin = coincidencias[i + 1].start() if i + 1 < len(coincidencias) else len(texto)
        fragmento = texto[inicio:fin].strip()
        bloques.append((inicio, fin, _extraer_articulo(fragmento)))

    return bloques


def _buscar_corte_interno(texto: str, inicio: int, fin: int) -> int | None:
    """
    Busca un corte natural dentro de un mismo bloque/artículo.

    Si un artículo supera el límite del embedding, se parte por parágrafo,
    párrafo o fin de oración, pero nunca se pierde la referencia del artículo.
    """
    minimo = inicio + (fin - inicio) // 3

    mejor = None
    for match in _PATRON_PARAGRAFO.finditer(texto, inicio, fin):
        if match.start() > minimo:
            mejor = match.start()
    if mejor is not None:
        return mejor

    corte = texto.rfind("\n\n", inicio, fin)
    if corte > minimo:
        return corte

    corte = texto.rfind(". ", inicio, fin)
    if corte > minimo:
        return corte + 1

    return None


# ─── Corte de límite de artículo ─────────────────────────────────────────────

def _buscar_corte_articulo(texto: str, inicio: int, fin: int) -> int | None:
    """
    Busca el inicio de un nuevo artículo, capítulo o parágrafo
    dentro del rango [inicio, fin] del texto.

    Prioridad: ARTÍCULO > CAPÍTULO > PARÁGRAFO

    El corte debe estar a más de 1/3 del inicio para evitar
    chunks demasiado pequeños.

    Args:
        texto:  Texto completo.
        inicio: Posición de inicio del chunk actual.
        fin:    Límite máximo del chunk.

    Returns:
        Posición del corte si se encontró, None si no.
    """
    minimo = inicio + (fin - inicio) // 3

    for patron in [_PATRON_ARTICULO, _PATRON_CAPITULO, _PATRON_PARAGRAFO]:
        # Buscar la ÚLTIMA ocurrencia dentro del rango (corte más tardío posible)
        mejor = None
        for m in patron.finditer(texto, inicio, fin):
            if m.start() > minimo:
                mejor = m.start()
        if mejor is not None:
            return mejor

    return None


# ─── División en chunks ───────────────────────────────────────────────────────

def dividir_en_chunks(
    texto:         str,
    nombre_fuente: str,
    tipo_doc:      str = "otro",
    tamanio_chunk: int = LIMITE_CARACTERES,
    overlap:       int = OVERLAP_CARACTERES
) -> list[dict]:
    """
    Divide el texto en chunks con overlap inteligente.

    Prioridad de corte (de mayor a menor):
      1. Inicio de nuevo ARTÍCULO
      2. Inicio de nuevo CAPÍTULO
      3. Inicio de PARÁGRAFO
      4. Párrafo doble (\\n\\n)
      5. Punto seguido de espacio (. )
      6. Corte duro en el límite de caracteres

    Cada chunk incluye:
      - chunk_id:    ID único basado en fuente + número
      - fuente:      Nombre del PDF original
      - pagina:      Página donde inicia el chunk
      - texto:       Contenido del chunk
      - articulo:    Artículo más cercano al inicio del chunk
      - tipo_doc:    "RIT", "BPM" u "otro"
      - char_inicio: Posición de inicio en el texto limpio
      - char_fin:    Posición de fin en el texto limpio

    Args:
        texto:         Texto del .txt (sin marcador [TIPO_DOC]).
        nombre_fuente: Nombre del PDF original.
        tipo_doc:      Tipo detectado por el loader.
        tamanio_chunk: Máximo de caracteres por chunk.
        overlap:       Caracteres repetidos entre chunks consecutivos.

    Returns:
        Lista de dicts con la estructura descrita arriba.
    """
    texto_limpio, mapa_paginas = limpiar_texto(texto)

    # Limpieza final: reducir líneas vacías excesivas
    texto_limpio = re.sub(r'\n{3,}', '\n\n', texto_limpio).strip()

    chunks   = []
    chunk_id = 1

    for inicio_bloque, fin_bloque, articulo_bloque in _segmentar_por_articulos(texto_limpio):
        texto_bloque = texto_limpio[inicio_bloque:fin_bloque].strip()
        if not texto_bloque:
            continue

        inicio_rel = 0
        total_bloque = len(texto_bloque)

        while inicio_rel < total_bloque:
            fin_rel = min(inicio_rel + tamanio_chunk, total_bloque)

            if fin_rel < total_bloque:
                corte = _buscar_corte_interno(texto_bloque, inicio_rel, fin_rel)
                if corte is not None:
                    fin_rel = corte

            fragmento = texto_bloque[inicio_rel:fin_rel].strip()
            if not fragmento:
                inicio_rel = fin_rel
                continue

            inicio_abs = inicio_bloque + inicio_rel
            fin_abs    = inicio_bloque + fin_rel
            pagina     = obtener_pagina_en_posicion(inicio_abs, mapa_paginas)
            articulo   = articulo_bloque or _extraer_articulo(fragmento)

            chunks.append({
                "chunk_id":    f"{Path(nombre_fuente).stem}_{chunk_id:03d}",
                "fuente":      nombre_fuente,
                "pagina":      pagina,
                "texto":       fragmento,
                "articulo":    articulo,
                "tipo_doc":    tipo_doc,
                "char_inicio": inicio_abs,
                "char_fin":    fin_abs
            })
            chunk_id += 1

            siguiente = fin_rel - overlap
            inicio_rel = siguiente if siguiente > inicio_rel else fin_rel

    return chunks


# ─── Pipeline de procesamiento ────────────────────────────────────────────────

def procesar_textos(directorio_processed: str, directorio_chunks: str) -> list[dict]:
    """
    Procesa todos los .txt en /processed y genera los .json en /chunks.

    Lee el marcador [TIPO_DOC: X] de cada .txt para propagar el tipo
    de documento al metadata de cada chunk.

    Args:
        directorio_processed: Ruta a la carpeta con los .txt.
        directorio_chunks:    Ruta a la carpeta de salida de chunks.

    Returns:
        Lista con resumen de procesamiento por archivo.
    """
    processed_path = Path(directorio_processed)
    chunks_path    = Path(directorio_chunks)

    chunks_path.mkdir(parents=True, exist_ok=True)

    archivos_txt = list(processed_path.glob("*.txt"))

    if not archivos_txt:
        print("⚠️  No se encontraron archivos .txt en /processed")
        return []

    print(f"📂 Se encontraron {len(archivos_txt)} archivo(s) en /processed\n")

    resumen = []

    for archivo_txt in archivos_txt:
        print(f"  ✂️  Procesando: {archivo_txt.name}")

        with open(archivo_txt, "r", encoding="utf-8") as f:
            texto = f.read()

        # Leer tipo_doc del marcador inyectado por loader.py
        tipo_doc, texto_sin_marcador = _extraer_tipo_doc(texto)
        print(f"     Tipo: {tipo_doc}")

        nombre_fuente = archivo_txt.stem + ".pdf"

        chunks = dividir_en_chunks(
            texto=         texto_sin_marcador,
            nombre_fuente= nombre_fuente,
            tipo_doc=      tipo_doc,
            tamanio_chunk= LIMITE_CARACTERES,
            overlap=       OVERLAP_CARACTERES
        )

        # Verificar límite de caracteres
        chunks_largos = [c for c in chunks if len(c["texto"]) > LIMITE_CARACTERES]
        if chunks_largos:
            print(f"  ⚠️  {len(chunks_largos)} chunk(s) superan {LIMITE_CARACTERES} chars")
        else:
            print(f"  ✅ Todos los chunks dentro del límite ({LIMITE_CARACTERES} chars)")

        # Mostrar muestra de artículos detectados
        arts = [c["articulo"] for c in chunks if c["articulo"]]
        if arts:
            print(f"     Artículos detectados: {len(arts)} / {len(chunks)} chunks")
            print(f"     Ejemplo: {arts[0]}")

        nombre_salida = archivo_txt.stem + ".json"
        ruta_salida   = chunks_path / nombre_salida

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"  ✅ {len(chunks)} chunks → /chunks/{nombre_salida}\n")

        resumen.append({
            "archivo":  archivo_txt.name,
            "chunks":   len(chunks),
            "tipo_doc": tipo_doc,
            "estado":   "ok"
        })

    return resumen


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    resumen = procesar_textos(
        directorio_processed= str(BASE_DIR / "processed"),
        directorio_chunks=    str(BASE_DIR / "chunks")
    )

    print("=" * 55)
    print("RESUMEN DE CHUNKS")
    print("=" * 55)
    for r in resumen:
        print(f"✅ {r['archivo']:<40} {r['chunks']:>4} chunks  "
              f"tipo: {r['tipo_doc']}")
