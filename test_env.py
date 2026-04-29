import sys

sys.path.insert(0, "src")

import ollama
from qdrant_client import QdrantClient
from console import configurar_salida_utf8

configurar_salida_utf8()

print("Probando Ollama embeddings...")
resp = ollama.embeddings(model="mxbai-embed-large", prompt="test de conexión")
print(f"✅ Ollama OK — dimensiones del vector: {len(resp['embedding'])}")

print("\nProbando Qdrant...")
client = QdrantClient(url="http://localhost:6333")
colecciones = client.get_collections()
print(f"✅ Qdrant OK — colecciones existentes: {colecciones}")

print("\n🎉 Entorno listo para el Paso 2")
