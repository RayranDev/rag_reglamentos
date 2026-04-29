import os
import json
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from classifier import clasificar_pregunta
from retriever import buscar_chunks_relevantes, formatear_contexto
from llm import generar_respuesta, respuesta_no_encontrada

app = FastAPI(title="RAG Plastitec API")

# Semáforo para serializar peticiones a Ollama y evitar OOM
ollama_semaphore = asyncio.Semaphore(1)

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
API_LOG_FILE = LOGS_DIR / "api_queries.log"

DATA_DIR = BASE_DIR / "data"
FAQS_FILE = DATA_DIR / "faqs.json"

# Modelos Pydantic
class AskRequest(BaseModel):
    pregunta: str
    skip_faq_increment: bool = False

class FAQIncrementRequest(BaseModel):
    id_faq: int

def get_faqs():
    if FAQS_FILE.exists():
        with open(FAQS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_faqs(faqs):
    with open(FAQS_FILE, "w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)

def log_query(log_data: dict):
    with open(API_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

def extract_fuente_confianza(texto_respuesta: str):
    lineas = texto_respuesta.strip().split("\n")
    fuente = "No aplica"
    confianza = "baja"
    respuesta_limpia = []
    
    for linea in lineas:
        l_lower = linea.lower().strip()
        if l_lower.startswith("fuente:"):
            fuente = linea.split(":", 1)[1].strip()
        elif l_lower.startswith("confianza:"):
            confianza = linea.split(":", 1)[1].strip()
        else:
            respuesta_limpia.append(linea)
            
    # Reconstruir la respuesta sin esas líneas al final
    texto = "\n".join(respuesta_limpia).strip()
    return texto, fuente, confianza

def check_faq_match(pregunta: str):
    pregunta_lower = pregunta.lower().strip()
    faqs = get_faqs()
    for faq in faqs:
        # Similitud básica
        if faq["pregunta"].lower().strip() in pregunta_lower or pregunta_lower in faq["pregunta"].lower().strip():
            faq["frecuencia"] += 1
            save_faqs(faqs)
            break

@app.post("/ask")
async def ask_endpoint(req: AskRequest, request: Request):
    inicio = time.time()
    pregunta = req.pregunta
    user_agent = request.headers.get("user-agent", "unknown")
    
    # 1. Verificar coincidencia con FAQ internamente (solo si no viene del botón)
    if not req.skip_faq_increment:
        check_faq_match(pregunta)
    
    # 2. Clasificar la pregunta
    clasificacion = clasificar_pregunta(pregunta)
    
    if clasificacion["resultado"] == "SENSIBLE":
        tiempo_total = time.time() - inicio
        respuesta_texto = "Para este tipo de consultas, por favor dirígete al área de Recursos Humanos, donde podrán brindarte una orientación más personalizada."
        log_query({
            "timestamp": time.time(),
            "pregunta": pregunta,
            "confianza": "alta",
            "tiempo_total": round(tiempo_total, 2),
            "bloqueado": True,
            "user_agent": user_agent
        })
        return {
            "respuesta": respuesta_texto,
            "fuente": "Políticas RRHH",
            "confianza": "alta",
            "tiempo_total": round(tiempo_total, 2),
            "bloqueado": True
        }
        
    # 3. Buscar chunks
    chunks = buscar_chunks_relevantes(pregunta)
    if not chunks:
        tiempo_total = time.time() - inicio
        resp_dict = respuesta_no_encontrada()
        resp, fte, conf = extract_fuente_confianza(resp_dict["respuesta"])
        log_query({
            "timestamp": time.time(),
            "pregunta": pregunta,
            "confianza": conf,
            "tiempo_total": round(tiempo_total, 2),
            "bloqueado": False,
            "user_agent": user_agent
        })
        return {
            "respuesta": resp,
            "fuente": fte,
            "confianza": conf,
            "tiempo_total": round(tiempo_total, 2),
            "bloqueado": False
        }
        
    # 4. Generar respuesta (con control de concurrencia usando semáforo)
    contexto = formatear_contexto(chunks)
    
    # Ollama generation runs in thread to avoid blocking loop, but gated by semaphore
    async with ollama_semaphore:
        # Timeout at 30 seconds
        try:
            resp_dict = await asyncio.wait_for(
                asyncio.to_thread(generar_respuesta, contexto, pregunta), 
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="El modelo LLM no respondió a tiempo.")

    tiempo_total = time.time() - inicio
    
    # Parsear respuesta para separar fuente y confianza
    resp, fte, conf = extract_fuente_confianza(resp_dict["respuesta"])
    
    log_query({
        "timestamp": time.time(),
        "pregunta": pregunta,
        "confianza": conf,
        "tiempo_total": round(tiempo_total, 2),
        "bloqueado": False,
        "user_agent": user_agent
    })

    return {
        "respuesta": resp,
        "fuente": fte,
        "confianza": conf,
        "tiempo_total": round(tiempo_total, 2),
        "bloqueado": False
    }

@app.get("/faq")
async def get_faqs_endpoint():
    faqs = get_faqs()
    # Ordenar por frecuencia descendente y retornar Top 5
    faqs_ordenadas = sorted(faqs, key=lambda x: x["frecuencia"], reverse=True)
    return faqs_ordenadas[:5]

@app.post("/faq/increment")
async def increment_faq(req: FAQIncrementRequest):
    faqs = get_faqs()
    for faq in faqs:
        if faq["id"] == req.id_faq:
            faq["frecuencia"] += 1
            save_faqs(faqs)
            return {"success": True, "faq_id": faq["id"], "frecuencia_nueva": faq["frecuencia"]}
    raise HTTPException(status_code=404, detail="FAQ no encontrada")

# Servir estáticos
WEB_UI_DIR = BASE_DIR / "src" / "web_ui"
WEB_UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(WEB_UI_DIR), html=True), name="web_ui")

