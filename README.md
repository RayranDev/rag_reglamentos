# RAG Reglamentos Plastitec

Sistema RAG local para consultar el Reglamento Interno de Trabajo (RIT) y
documentos corporativos/BPM de Plastitec.

Estado documentado: 2026-04-28.

## Estado Actual

El sistema esta funcionando con ingesta completa, busqueda hibrida, evaluacion
masiva y reporte de consumo de maquina.

Ultima evaluacion completa:

- Archivo: `evaluation/resultados_RAG_20260428_091807.csv`
- Resumen maquina: `evaluation/resumen_maquina_20260428_091807.json`
- Preguntas evaluadas: `153`
- Resultados: `149 alta`, `2 media`, `2 baja`
- Calidad alta: `97.4%`
- Calidad alta + media: `98.7%`
- Tiempo promedio: `3.35s` por pregunta
- Tiempo maximo: `9.17s`

Documentacion detallada:

- [Estado tecnico del RAG](docs/estado_tecnico_rag.md)

## Arquitectura

Flujo principal:

```text
PDFs -> texto procesado -> chunks por articulo -> embeddings Ollama
     -> Qdrant -> retriever hibrido -> LLM -> respuesta con fuente
```

Componentes principales:

- `ingest.py`: procesa PDFs, genera chunks, crea embeddings e indexa en Qdrant.
- `src/loader.py`: extrae texto de PDFs y detecta tipo de documento.
- `src/chunker.py`: divide documentos, priorizando articulos normativos.
- `src/embeddings.py`: genera embeddings con Ollama.
- `src/vector_db.py`: administra coleccion, indices e insercion en Qdrant.
- `src/retriever.py`: busqueda hibrida con embeddings, filtros y keywords curadas.
- `src/classifier.py`: bloquea consultas sensibles antes de enviarlas al RAG.
- `src/llm.py`: genera respuestas usando solo el contexto recuperado.
- `src/main.py`: interfaz de consulta por consola.
- `evaluar_rit.py`: evaluacion masiva y reporte de recursos.

## Requisitos

Requisitos de software:

- Python 3.12
- Docker
- Ollama local
- Qdrant por Docker Compose
- Modelo LLM: `llama3`
- Modelo embeddings: `mxbai-embed-large`

Requisitos de maquina recomendados para este volumen:

- RAM: `16 GB` minimo recomendado, `32 GB` ideal
- GPU: `8 GB VRAM` recomendado para tiempos similares a la evaluacion
- Disco libre: `15 GB` minimo practico, `20 GB+` recomendado
- SSD recomendado

Modelos usados en la corrida:

- `llama3:latest`: `4.7 GB`
- `mxbai-embed-large:latest`: `669 MB`

## Configuracion

Variables principales en `.env`:

```env
LLM_MODEL=llama3
EMBED_MODEL=mxbai-embed-large
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=reglamentos
TOP_K=8
SCORE_THRESHOLD=0.60
```

Instalar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Levantar Qdrant:

```powershell
docker-compose up -d
```

Verificar entorno:

```powershell
.\.venv\Scripts\python.exe test_env.py
```

## Ingesta

Coloca los PDFs en `data/` y ejecuta:

```powershell
.\.venv\Scripts\python.exe ingest.py
```

La ingesta:

- Extrae texto a `processed/`.
- Genera chunks en `chunks/`.
- Crea embeddings con Ollama.
- Crea la coleccion `reglamentos` en Qdrant.
- Crea indices de payload.
- Inserta vectores con metadata de fuente, pagina, articulo y tipo de documento.

Estado actual de ingesta:

- PDFs procesados: `10`
- Chunks totales: `429`
- Vectores en Qdrant: `429`
- Chunks del RIT: `354`
- Chunks BPM: `75`

## Consulta

```powershell
.\.venv\Scripts\python.exe src\main.py
```

El flujo de consulta:

1. Clasifica la pregunta y bloquea temas sensibles.
2. Recupera chunks relevantes con busqueda semantica + keywords.
3. Formatea contexto con fuente, pagina y articulo.
4. Genera respuesta con el LLM.
5. Devuelve respuesta, fuente y confianza.

## Evaluacion

Ejecutar evaluacion completa:

```powershell
.\.venv\Scripts\python.exe evaluar_rit.py
```

Ejecutar corrida corta:

```powershell
$env:EVAL_LIMIT='5'
.\.venv\Scripts\python.exe evaluar_rit.py
Remove-Item Env:\EVAL_LIMIT
```

La evaluacion genera:

- `evaluation/resultados_RAG_*.csv`: respuesta, confianza, chunks usados,
  tiempos por etapa y metricas por pregunta.
- `evaluation/resumen_maquina_*.json`: tiempos agregados, CPU, RAM, disco,
  GPU/VRAM y almacenamiento usado.

## Pruebas Utiles

```powershell
.\.venv\Scripts\python.exe test_classifier.py
.\.venv\Scripts\python.exe test_retriever_quality.py
.\.venv\Scripts\python.exe test_busqueda.py
.\.venv\Scripts\python.exe -m compileall src ingest.py evaluar_rit.py
```

## Pendientes Conocidos

La ultima evaluacion dejo 4 casos no perfectos:

- Media: uso de celular personal durante jornada.
- Media: jefe pidiendo trabajar mas horas que lo permitido por ley.
- Baja: cierre intempestivo de la empresa.
- Baja: beneficios generales de trabajar en Plastitec segun el RIT.

Estos pendientes parecen puntuales y no estructurales. Dos dependen de reglas
del RIT que se pueden reforzar en retrieval; los otros dos son preguntas mas
ambiguas o con cobertura documental menos directa.
