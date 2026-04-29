# PROMPT ÚNICO – FASES 2 Y 3 DEL SISTEMA RAG PLASTITEC

Actúa como un arquitecto de software senior experto en Python, sistemas RAG locales, Ollama, FastAPI y HTML/CSS/JS vanilla sin frameworks pesados.

Vas a implementar las **Fases 2 y 3** del sistema RAG **Plastitec** partiendo de un MVP de consola ya funcional con las siguientes características confirmadas:

- 100% offline: Python + Ollama + Qdrant (Docker local)
- Modelos: `llama3` (generación), `mxbai-embed-large` (embeddings)
- 10 PDFs ingestados → 429 chunks (354 RIT, 75 BPM), chunker por artículos
- Recuperación híbrida (embeddings + keywords + filtros por tipo de documento)
- Clasificador de preguntas sensibles (redirige a RRHH)
- Temperatura 0.18, formato obligatorio `Fuente:` y `Confianza:`, reglas determinísticas
- Métricas: 97.4% respuestas confianza alta, promedio 3.35 s, VRAM ~6 GB
- Estructura de carpetas:

```
/project
├── data/              # PDFs originales
├── processed/         # texto limpio
├── chunks/            # chunks JSON
├── vectorstore/       # Qdrant persistencia
├── src/               # loader, chunker, embeddings, vector_db, retriever,
│                        classifier, llm, main.py
├── evaluation/        # CSVs, JSONs de evaluación
├── docker-compose.yml
└── ingest.py, evaluar_rit.py, etc.
```

No rediseñes lo que ya funciona. Extiéndelo con control sin romper la estabilidad alcanzada.

---

## FASE 2 – API REST + INTERFAZ WEB

### 1. `src/api.py` (nuevo)

Crea una aplicación FastAPI con los siguientes endpoints:

- `POST /ask` → recibe `{"pregunta": "..."}`, ejecuta el pipeline RAG existente y devuelve:
```json
{
  "respuesta": "texto",
  "fuente": "Artículo X...",
  "confianza": "alta",
  "tiempo_total": 3.2,
  "bloqueado": false
}
```
- `GET /faq` → devuelve el Top 5 de preguntas frecuentes ordenadas por frecuencia.
- `POST /faq/increment` → recibe `{"id_faq": 1}` e incrementa el contador de esa FAQ.

Requisitos adicionales de la API:
- Control de concurrencia con semáforo (Ollama no es thread-safe).
- Timeout de 30 s por solicitud.
- Logging de cada consulta en `logs/api_queries.log` (JSON por línea) con: timestamp, pregunta, confianza, tiempo total, bloqueado, user-agent.
- Al recibir una pregunta, verificar internamente si coincide con alguna FAQ (por texto simple o embedding) e incrementar su contador si aplica.

### 2. `data/faqs.json`

Crear este archivo si no existe, con las 5 preguntas curadas del dominio Plastitec:

```json
[
  {"id": 1, "pregunta": "¿Cómo solicito vacaciones?", "frecuencia": 1},
  {"id": 2, "pregunta": "¿Cuál es el horario de trabajo?", "frecuencia": 1},
  {"id": 3, "pregunta": "¿Cómo reporto un accidente laboral?", "frecuencia": 1},
  {"id": 4, "pregunta": "¿Puedo usar mi celular en la planta?", "frecuencia": 1},
  {"id": 5, "pregunta": "¿Cuántos días de permiso por luto tengo?", "frecuencia": 1}
]
```

El endpoint `GET /faq` devuelve siempre las 5 con mayor frecuencia, ordenadas de mayor a menor.

### 3. `src/web_ui/` (carpeta nueva)

Crea tres archivos completamente funcionales:

**`index.html`** – Estructura semántica accesible con:
- Cabecera: `Tu Asistente de RRHH - Oficial`
- Área de chat con scroll automático al recibir respuesta
- Sección "Top 5 Preguntas Frecuentes (Respuestas Visuales)" con botones clicables
- Indicador "🎤 Escuchando..." visualmente inactivo (opacidad 0.5)
- Botón "¡Toca para hablar!" deshabilitado con tooltip "Próximamente"
- Pie de página con texto exacto: `Sistema de Consulta y Respuestas de RRHH | Procesamiento Local Offline con Llama 3 | Interfaz Web Adaptable (Tablet/Móvil)`

**`style.css`** – Estilos responsivos con:
- Burbujas de chat diferenciadas: usuario a la derecha (azul), asistente a la izquierda (gris claro)
- Paleta corporativa: azul marino, gris, blanco
- Media queries para 375px (móvil), 768px (tablet), 1024px (desktop)
- Clase `.disabled` con `opacity: 0.5; cursor: not-allowed` para elementos de voz

**`script.js`** – Lógica con:
- `fetch POST /ask` al enviar una pregunta; mostrar respuesta en burbuja con la fuente al final (`Fuente: Artículo X...`)
- `fetch GET /faq` al cargar la página; renderizar los 5 botones dinámicamente
- Al pulsar una FAQ: insertar la pregunta en el chat + `fetch POST /faq/increment` con el id correspondiente
- Scroll automático al final del chat tras cada respuesta
- Botón de micrófono y texto "Escuchando..." deshabilitados (sin funcionalidad activa en Fase 2)

Servir toda la carpeta `src/web_ui/` como archivos estáticos desde FastAPI con `StaticFiles`.

### 4. Modificación de `main.py`

Agregar modo servidor:
```bash
python main.py --server   # levanta la API en http://localhost:8000
python main.py            # modo consola sin cambios
```

### 5. Script `evaluar_api.py`

Reproducir la batería de 153 preguntas contra el endpoint `POST /ask` y verificar que se mantiene ≥97% de respuestas con confianza alta. Registrar latencias y desglosarlas por retrieval y LLM.

---

## FASE 3 – ENTRADA Y SALIDA POR VOZ

Implementar esta fase solo cuando la Fase 2 esté estable y las métricas de evaluación no hayan empeorado.

### Componentes

- **STT:** `faster-whisper` con modelo `small` o `medium` (elegir según consumo de VRAM disponible)
- **TTS:** `pyttsx3` como primera opción (offline, sin GPU); evaluar Coqui TTS si se requiere mayor calidad

### Nuevo endpoint: `POST /ask/voice`

- Recibe un archivo de audio (WAV o MP3)
- Transcribe con Whisper → ejecuta el pipeline RAG → sintetiza respuesta con TTS
- Devuelve: texto de la respuesta + archivo de audio opcional

### Gestión de VRAM

La VRAM disponible (~8 GB) ya está casi saturada con Llama3 (~6 GB). Implementar un `ModelManager` que:
- Descargue Llama3 de VRAM cuando no se esté usando
- Cargue Whisper y TTS bajo demanda
- Documente el aumento de latencia resultante

### Activación de la UI de voz

Cuando el endpoint `/ask/voice` esté operativo:
- El indicador "🎤 Escuchando..." se activará con texto dinámico (`Escuchando...` / `Procesando...`)
- El botón "¡Toca para hablar!" se habilitará y usará la API `MediaRecorder` del navegador para capturar audio
- Al terminar la grabación: enviar audio a `/ask/voice`, mostrar burbuja de texto y reproducir audio si se devuelve

---

## REGLAS DE DESARROLLO

1. **Iteración pequeña:** primero la API sin interfaz; probar con `curl` antes de integrar la UI.
2. **No romper lo actual:** ejecutar `evaluar_rit.py` antes y después de cada módulo nuevo.
3. **Documentar:** agregar comentarios en el código y actualizar `CHANGELOG.md` por cada módulo modificado.
4. **Concurrencia:** no asumir que Ollama es thread-safe; implementar cola serializada.
5. **Fidelidad visual:** la UI debe coincidir con el diseño de referencia en estructura y disposición. Paleta profesional: azul marino, gris, blanco.
6. **Idioma:** todos los textos de la interfaz en español.
7. **Responsive obligatorio:** verificar en 375px, 768px y 1024px.

---

## CRITERIOS DE ÉXITO

### Fase 2 lista cuando:
- `POST /ask` responde en ≤ 4 s promedio
- `evaluar_api.py` reporta ≥ 97% confianza alta
- La cola maneja 3 solicitudes concurrentes sin errores
- La interfaz carga FAQs dinámicamente y el clic en una FAQ envía la pregunta al chat
- El contador de frecuencia se actualiza y reordena el Top 5
- Los elementos de voz están presentes pero deshabilitados
- El pie de página dice exactamente: `Procesamiento Local Offline con Llama 3`
- La interfaz es responsive en los tres viewports

### Fase 3 lista cuando:
- El flujo de voz funciona 100% offline con latencia adicional < 2 s
- La VRAM no excede el límite físico (con swap de modelos documentado si aplica)
- La calidad de transcripción es aceptable para preguntas típicas de RIT/BPM

---

## ENTREGABLES ESPERADOS

### Al finalizar Fase 2:
- `src/api.py`
- `src/web_ui/index.html`, `style.css`, `script.js`
- `evaluar_api.py`
- `data/faqs.json` precargado
- `main.py` actualizado con flag `--server`
- `README.md` con instrucciones de arranque

### Al finalizar Fase 3:
- Módulos STT/TTS integrados
- Endpoint `POST /ask/voice`
- UI de voz activada
- Reporte de consumo de recursos actualizado

---

Comienza implementando `src/api.py` y verifica que responde correctamente con `curl` antes de continuar con la interfaz web.
