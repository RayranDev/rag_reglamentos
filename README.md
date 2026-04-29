# RAG Plastitec - Asistente IA Offline de Recursos Humanos

Sistema RAG (Retrieval-Augmented Generation) especializado en responder consultas sobre reglamentos internos y políticas de Recursos Humanos de la empresa Plastitec. Diseñado para operar **100% de manera local (offline)** con el fin de proteger la privacidad de los datos, incluye además capacidades de reconocimiento y síntesis de voz gestionando inteligentemente los recursos de hardware.

## 🚀 Características Principales

*   **Búsqueda Semántica Rápida:** Utiliza bases de datos vectoriales para encontrar el contexto adecuado independientemente de si el usuario usa sinónimos.
*   **Modelo de Lenguaje Local:** Emplea Llama 3 (a través de Ollama) garantizando la privacidad absoluta.
*   **Filtro de Seguridad (Clasificador):** Intercepta preguntas sensibles (acoso, violencia, quejas graves) antes de que lleguen al modelo y las redirige al personal de Recursos Humanos.
*   **Voz a Texto (STT) y Texto a Voz (TTS):** Permite hacer preguntas hablando por el micrófono y responde con audio, todo en el mismo servidor local sin usar APIs de terceros (Whisper + pyttsx3).
*   **Gestión Inteligente de VRAM:** Descarga el modelo de lenguaje de la tarjeta gráfica automáticamente al usar el modelo de reconocimiento de voz para evitar colapsos por falta de memoria (Out of Memory).
*   **Control de Concurrencia:** Semáforo integrado para serializar peticiones, evitando cuellos de botella en equipos de bajos recursos.
*   **Interfaz Web Responsiva:** Interfaz estilo chat de uso amigable con sugerencias interactivas de las "Preguntas Frecuentes" más populares.

---

## 🛠️ Stack Tecnológico

*   **Backend:** Python 3.10+ / FastAPI / Uvicorn
*   **Base de Datos Vectorial:** Qdrant (Local en disco)
*   **Modelos de Embeddings:** HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
*   **LLM (Generación):** Llama 3 (8B) gestionado por Ollama
*   **STT (Transcripción):** `faster-whisper` (modelo: *small*)
*   **TTS (Síntesis de voz):** `pyttsx3`
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript

---

## 📋 Requisitos Previos

1.  **Hardware:**
    *   Procesador moderno multicore.
    *   RAM del sistema: Mínimo 16 GB.
    *   **GPU VRAM:** Mínimo 6 GB (NVIDIA preferible) para evitar latencias altas y cuellos de botella.
2.  **Software:**
    *   [Python 3.10+](https://www.python.org/downloads/)
    *   [Ollama](https://ollama.com/) instalado y ejecutándose en segundo plano (`http://localhost:11434`).

---

## ⚙️ Instalación y Configuración

1.  **Clonar el repositorio y entrar al directorio:**
    ```bash
    git clone <url-del-repo>
    cd rag_reglamentos
    ```

2.  **Crear y activar un entorno virtual (recomendado):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar el modelo en Ollama:**
    ```bash
    ollama run llama3
    ```

---

## 🏃‍♂️ Cómo Usar el Sistema

### 1. Ingesta de Documentos (Base de conocimiento)
Antes de poder preguntar, necesitas alimentar el sistema con los reglamentos o manuales de la empresa:
1. Coloca tus archivos en la carpeta `data/`. El sistema soporta una gran variedad de formatos nativamente (`.pdf`, `.txt`, `.md`, entre otros).
2. Ejecuta el script de ingesta:
   ```bash
   python src/ingest.py
   ```
*Nota: Este proceso creará fragmentos inteligentes (chunks) respetando los artículos legales y construirá la base de datos vectorial en la carpeta `vectorstore/`.*

### 2. Levantar el Servidor Completo (API + Web UI)
Para ejecutar la aplicación con la interfaz web y los endpoints de voz:
```bash
python src/main.py --server
```
* Accede a la interfaz web en tu navegador: `http://localhost:8000`
* Accede a la documentación automática de la API en: `http://localhost:8000/docs`

### 3. Modo Consola Clásico (Para pruebas rápidas)
Si solo deseas interactuar en la terminal sin levantar el servidor web:
```bash
python src/main.py
```

---

## 📡 API Endpoints

El sistema expone las siguientes rutas en el puerto `8000`:

*   `POST /ask`: Recibe `{ "pregunta": "texto...", "skip_faq_increment": false }`. Devuelve la respuesta RAG estructurada y fuentes.
*   `POST /ask/voice`: Recibe un archivo `UploadFile` (audio en `.webm` u otro formato). Devuelve la respuesta en texto y un string en **Base64** con el archivo `.wav` de la voz generada.
*   `GET /faq`: Retorna el top 5 de las preguntas más frecuentes almacenadas en `data/faqs.json`.
*   `POST /faq/increment`: Incrementa la popularidad de una pregunta frecuente al hacerle clic.
*   `GET /`: Sirve la interfaz web estática (`index.html`).

---

## 🧠 Gestión de Hardware (VRAM Swap)
Al invocar el endpoint `/ask/voice`, la clase `VoiceManager` realiza una estrategia de *"Swap"* para cuidar tu GPU:
1. Pide a Ollama que descargue Llama 3 (`keep_alive=0`).
2. Carga `faster-whisper` a la VRAM.
3. Transcribe el audio del usuario.
4. Elimina `faster-whisper` de la VRAM y vacía el caché (`torch.cuda.empty_cache()`).
5. Ollama recarga Llama 3 automáticamente al generar la respuesta.

Esta operación dura ~2 segundos pero asegura que el equipo no crashee por falta de VRAM.

---

## 🗂️ Estructura de Directorios

```text
rag_reglamentos/
├── data/                  # Docs Markdown originales y faqs.json
├── chunks/                # Fragmentos JSON extraídos en la ingesta
├── vectorstore/           # Base de datos vectorial de Qdrant (Generada)
├── src/                   # Código fuente de la app
│   ├── api.py             # Rutas FastAPI y orquestación del servidor
│   ├── main.py            # Punto de entrada (Consola o Servidor)
│   ├── model_manager.py   # Gestor de modelos (Voz, VRAM, Ollama)
│   ├── chunker.py         # Lógica de partición inteligente de MD
│   ├── classifier.py      # Lógica del filtro de seguridad
│   ├── llm.py             # Conexión con Ollama y Prompts
│   ├── retriever.py       # Lógica de búsqueda vectorial en Qdrant
│   └── web_ui/            # Interfaz gráfica estática (HTML/CSS/JS)
├── prompt/                # Prompts utilizados para el sistema
├── evaluar_api.py         # Batería de pruebas automatizadas contra la API
├── evaluar_rit.py         # Framework de evaluación RAG Offline
├── requirements.txt       # Dependencias del backend
└── README.md              # Este archivo
```
