import os
import gc
import requests
import pyttsx3
import threading
from pathlib import Path

# Intentar importar torch y faster_whisper
try:
    import torch
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# Lock global para TTS (pyttsx3 no es thread-safe)
tts_lock = threading.Lock()

class VoiceManager:
    def __init__(self, llm_model="llama3:8b", whisper_size="small"):
        self.llm_model = llm_model
        self.whisper_size = whisper_size
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def descargar_llm(self):
        """Obliga a Ollama a descargar el modelo de la VRAM."""
        print(f"  [VoiceManager] Solicitando descarga de {self.llm_model} de VRAM...")
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {"model": self.llm_model, "keep_alive": 0}
            requests.post(url, json=payload, timeout=5)
            print("  [VoiceManager] Modelo LLM descargado.")
        except Exception as e:
            print(f"  [VoiceManager] Advertencia: No se pudo descargar LLM - {e}")

    def transcribir_audio(self, audio_path: str) -> str:
        """Carga Whisper, transcribe el audio y libera VRAM inmediatamente."""
        if not HAS_WHISPER:
            raise RuntimeError("faster-whisper no está instalado.")

        print(f"  [VoiceManager] Cargando Whisper ({self.whisper_size})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Cargar modelo
        model = WhisperModel(self.whisper_size, device=device, compute_type=compute_type)
        
        print("  [VoiceManager] Transcribiendo...")
        segments, info = model.transcribe(audio_path, language="es")
        texto = " ".join([segment.text for segment in segments])
        
        # Liberar memoria
        print("  [VoiceManager] Liberando Whisper de VRAM...")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return texto.strip()

    def generar_audio_tts(self, texto: str, output_path: str):
        """Genera un archivo de audio WAV a partir de texto usando pyttsx3."""
        print("  [VoiceManager] Generando TTS...")
        with tts_lock:
            engine = pyttsx3.init()
            # Opcional: configurar voz en español si hay varias disponibles
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.languages:
                    engine.setProperty('voice', voice.id)
                    break
                    
            engine.setProperty('rate', 155)  # Velocidad un poco más natural
            engine.save_to_file(texto, str(output_path))
            engine.runAndWait()
            # Importante destruir/limpiar el motor para evitar bugs en Windows
            del engine
