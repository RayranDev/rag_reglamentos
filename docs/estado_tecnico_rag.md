# Estado Tecnico del RAG Plastitec

Fecha de corte: 2026-04-28.

Este documento resume el estado actual del proyecto, las mejoras aplicadas,
las metricas de evaluacion y los criterios iniciales para pensar en produccion.

## 1. Objetivo

El proyecto implementa un asistente RAG local para responder preguntas sobre:

- Reglamento Interno de Trabajo de Plastitec.
- Instructivos corporativos/BPM.
- Documentos de salud, seguridad, etica e ingreso a areas de planta.

La respuesta debe basarse solo en documentos recuperados y debe incluir:

- Respuesta en lenguaje natural.
- `Fuente:`
- `Confianza: alta / media / baja`

## 2. Arquitectura

Flujo de extremo a extremo:

```text
data/*.pdf
  -> loader.py
  -> processed/*.txt
  -> chunker.py
  -> chunks/*.json
  -> embeddings.py
  -> Qdrant
  -> retriever.py
  -> llm.py
  -> respuesta final
```

Responsabilidades por modulo:

- `src/loader.py`: extrae texto de PDF, normaliza contenido y marca `tipo_doc`.
- `src/chunker.py`: divide el contenido. Para documentos normativos usa
  chunking por articulo como frontera principal.
- `src/embeddings.py`: genera embeddings con `mxbai-embed-large`.
- `src/vector_db.py`: crea coleccion, indices de payload e inserta vectores.
- `src/retriever.py`: recupera contexto con Qdrant, filtros y keywords curadas.
- `src/classifier.py`: detecta preguntas sensibles y evita respuestas riesgosas.
- `src/llm.py`: genera respuestas con `llama3` usando solo el contexto.
- `evaluar_rit.py`: ejecuta evaluacion masiva y mide consumo de maquina.

## 3. Configuracion Actual

Variables principales:

```env
LLM_MODEL=llama3
EMBED_MODEL=mxbai-embed-large
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=reglamentos
TOP_K=8
SCORE_THRESHOLD=0.60
```

Modelos locales relevantes:

| Modelo | Uso | Peso |
| --- | --- | ---: |
| `llama3:latest` | generacion de respuestas | 4.7 GB |
| `mxbai-embed-large:latest` | embeddings | 669 MB |

Tambien hay otros modelos instalados en la maquina, pero no forman parte del
flujo configurado actualmente.

## 4. Estado de Ingesta

Despues de reiniciar `processed/`, `chunks/`, `vectorstore/` y Qdrant, se
ejecuto una ingesta limpia con el chunker actualizado.

Resultado:

| Elemento | Cantidad |
| --- | ---: |
| PDFs procesados | 10 |
| Chunks totales | 429 |
| Vectores en Qdrant | 429 |
| Chunks RIT | 354 |
| Chunks BPM | 75 |

Distribucion de chunks:

| Archivo | Tipo | Chunks |
| --- | --- | ---: |
| `RIT PLASTITEC 25 NOV 2025.json` | RIT | 354 |
| `I-RH-004 - 14 Induccion en Seguridad Salud en el Trabajo.json` | BPM | 37 |
| `I-RH-003 - 18 BPMM (Material Visual) (1).json` | BPM | 10 |
| `I-RH-001-V- 20 -22 VERSION Conocimiento Plastitec...json` | BPM | 9 |
| `I-RH-017-5 CODIGO DE ETICA...json` | BPM | 7 |
| Otros instructivos BPM | BPM | 12 |

Qdrant:

- Coleccion: `reglamentos`
- Estado: `green`
- Total vectores: `429`
- Dimension esperada de embeddings: `1024`

## 5. Cambios Aplicados

### 5.1 Chunker por Articulos

Antes, el chunker partia principalmente por limite de caracteres. Esto rompia
articulos largos y dejaba muchos fragmentos sin metadata normativa.

Ahora:

- Se detectan encabezados `ARTICULO` / `ARTICULO` con y sin tilde.
- Se segmenta primero por articulo.
- Si un articulo supera el limite, se parte internamente por paragrafo,
  parrafo o punto.
- Todos los subchunks heredan el mismo `articulo`.
- El overlap se mantiene dentro del mismo articulo.

Impacto:

- El RIT paso a `354` chunks.
- Casi todos los chunks del RIT tienen referencia de articulo.
- El LLM recibe mejor fuente normativa y contexto mas coherente.

### 5.2 Retriever Hibrido

Se fortalecio `src/retriever.py`:

- Filtro automatico por `tipo_doc`: RIT o BPM.
- Cache de chunks en memoria.
- Query expansion solo cuando no hay keywords curadas.
- Keywords curadas para admision, aprendices, vacaciones, recargos,
  descansos, permisos, acoso, BPM y prohibiciones.
- Correccion de colisiones como `cargo` dentro de `recargo`.
- Refuerzo de casos especificos:
  - prueba de embarazo
  - libreta militar
  - aprendices ya vinculados
  - cuota minima de aprendices
  - fecha de vacaciones
  - recargos diurnos/nocturnos/dominicales
  - ingreso a area gris y vestidor
  - documentos de la empresa

### 5.3 LLM Menos Conservador

Se ajusto `src/llm.py`:

- Prompt mas claro: si el contexto contiene una regla aplicable, debe responder.
- Temperatura ajustada a `0.18`.
- Formato obligatorio con `Fuente:` y `Confianza:`.
- Reglas deterministicas para casos de alta precision cuando el texto
  recuperado contiene la formula normativa exacta.

Ejemplos cubiertos por reglas directas:

- Prueba de embarazo.
- Libreta militar.
- Vacaciones fijadas por Plastitec.
- Cuota minima de aprendices.
- Recargo extra diurno.
- Recargo dominical al 90% y 100%.
- Ingreso a vestidor/area gris.

### 5.4 Evaluacion con Recursos

`evaluar_rit.py` ahora reporta por pregunta:

- Tiempo total.
- Tiempo del clasificador.
- Tiempo del retriever.
- Tiempo del LLM.
- CPU/RAM del sistema.
- CPU/RAM del proceso Python.
- CPU/RAM aproximada de Ollama.
- Lectura/escritura de disco.
- GPU/VRAM si `nvidia-smi` esta disponible.
- Chunks usados y detalle JSON de chunks.

Al final genera un resumen agregado de maquina.

## 6. Evaluaciones Ejecutadas

### 6.1 Antes de las Ultimas Correcciones

Archivo: `evaluation/resultados_RAG_20260427_102819.csv`

| Confianza | Cantidad |
| --- | ---: |
| alta | 131 |
| media | 1 |
| baja | 21 |

Promedio reportado: `5.38s` por pregunta.

Problemas principales:

- Chunks sin articulo o con articulos partidos.
- Prompt demasiado conservador.
- Gaps y colisiones en `MAPA_KEYWORDS`.

### 6.2 Despues del Chunker y Primer Ajuste de Retrieval

Archivo: `evaluation/resultados_RAG_20260428_090047.csv`

| Confianza | Cantidad |
| --- | ---: |
| alta | 147 |
| media | 4 |
| baja | 2 |

Promedio: `3.52s` por pregunta.

Quedaban pendientes principalmente:

- Cuota de aprendices.
- Recargo dominical 90%/100%.
- Vestidor.
- Area gris.

### 6.3 Evaluacion Final Actual

Archivo: `evaluation/resultados_RAG_20260428_091807.csv`

| Confianza | Cantidad |
| --- | ---: |
| alta | 149 |
| media | 2 |
| baja | 2 |

Metricas:

| Metrica | Valor |
| --- | ---: |
| Preguntas procesadas | 153 |
| Tiempo total | 839.4s |
| Tiempo promedio | 3.35s |
| Tiempo minimo | 0.17s |
| Tiempo maximo | 9.17s |
| Promedio clasificador | 0.178s |
| Promedio retriever | 0.564s |
| Promedio LLM | 2.605s |

Lectura:

- `97.4%` de respuestas quedaron en confianza alta.
- `98.7%` quedaron en alta o media.
- El cuello principal ya no es retrieval; es la generacion con el LLM.

## 7. Consumo de Maquina

Fuente: `evaluation/resumen_maquina_20260428_091807.json`

Promedios:

| Recurso | Valor |
| --- | ---: |
| CPU sistema | 19.337% |
| RAM sistema | 50.659% |
| CPU Python | 1.31s |
| RAM Python | 87.678 MB |
| CPU Ollama | 3.133s |
| RAM Ollama | 1438.131 MB |
| GPU util | 0.0% |
| VRAM usada reportada | 5980 MB |
| VRAM total | 8151 MB |
| Disco usado | 22.412% |
| Lectura disco promedio | 1.028 MB |
| Escritura disco promedio | 2.889 MB |

Almacenamiento del proyecto:

| Carpeta | Tamano |
| --- | ---: |
| `data` | 14.56 MB |
| `processed` | 0.23 MB |
| `chunks` | 0.42 MB |
| `vectorstore` | 761.03 MB |
| `logs` | 1.85 MB |
| `evaluation` | 3.95 MB |

Notas:

- El vectorstore pesa mucho mas que los chunks por el almacenamiento interno de Qdrant.
- La VRAM reportada queda cerca de 6 GB con el modelo cargado.
- CPU promedio bajo indica que la maquina no esta saturada en esta corrida.
- El LLM es el principal costo de latencia por pregunta.

## 8. Requisitos Iniciales para Produccion Interna

Para mantener tiempos similares:

| Recurso | Minimo recomendado | Ideal |
| --- | ---: | ---: |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 12 GB |
| Disco libre | 15 GB | 20 GB+ |
| CPU | 4 nucleos modernos | 6-8 nucleos |
| Almacenamiento | SSD | NVMe |

Consideraciones:

- CPU-only probablemente funciona, pero con mayor latencia.
- Para varios usuarios concurrentes, el cuello sera Ollama/LLM.
- Si se requiere concurrencia real, conviene probar cola de solicitudes,
  limites por usuario o un servidor de inferencia dedicado.

## 9. Pendientes Conocidos

Ultima evaluacion:

| Nivel | Pregunta | Lectura |
| --- | --- | --- |
| media | Esta prohibido usar el celular personal durante la jornada laboral? | Falta evidencia documental directa o keyword especifica. |
| media | Puede mi jefe pedirme que trabaje mas horas de las permitidas por ley? | Recupera contexto no ideal; corregible con reglas de jornada/horas maximas. |
| baja | Que ocurre si el empleador cierra la empresa de manera intempestiva? | El articulo existe, pero retrieval no lo prioriza. |
| baja | Cuales son los beneficios generales de trabajar en PLASTITEC segun el RIT? | Pregunta amplia; el RIT no lista beneficios como seccion unica. |

Recomendacion:

1. Agregar keywords/reglas para cierre intempestivo y jornada maxima.
2. Decidir si "beneficios generales" debe responder desde prestaciones,
   vacaciones, licencias, dotacion y seguridad social, o si debe derivarse a RRHH.
3. Confirmar si existe politica explicita de celular personal. Si no existe,
   mantener respuesta conservadora.

## 10. Comandos Operativos

Levantar Qdrant:

```powershell
docker-compose up -d
```

Reingesta limpia manual:

```powershell
Remove-Item "chunks\*" -Recurse -Force
Remove-Item "processed\*" -Recurse -Force
Remove-Item "vectorstore\*" -Recurse -Force
docker-compose down
docker-compose up -d
.\.venv\Scripts\python.exe ingest.py
```

Consulta interactiva:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

Evaluacion completa:

```powershell
.\.venv\Scripts\python.exe evaluar_rit.py
```

Evaluacion corta:

```powershell
$env:EVAL_LIMIT='5'
.\.venv\Scripts\python.exe evaluar_rit.py
Remove-Item Env:\EVAL_LIMIT
```

Verificaciones:

```powershell
.\.venv\Scripts\python.exe test_env.py
.\.venv\Scripts\python.exe test_classifier.py
.\.venv\Scripts\python.exe test_retriever_quality.py
.\.venv\Scripts\python.exe -m compileall src ingest.py evaluar_rit.py
```

## 11. Criterio de Listo para Produccion

El sistema esta en buen estado para una prueba piloto interna si se acepta:

- Uso local con Ollama y Qdrant.
- Un volumen documental pequeno/medio.
- Revision humana de respuestas sensibles o ambiguas.
- Monitoreo de latencia, RAM y VRAM durante uso real.

Antes de produccion mas formal conviene:

- Resolver los 4 pendientes de evaluacion.
- Versionar una bateria fija de preguntas esperadas.
- Guardar snapshots de metricas por version del RAG.
- Definir politica de actualizacion documental e ingesta.
- Probar concurrencia con 2, 5 y 10 usuarios simulados.
