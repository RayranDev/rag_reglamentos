import sys
sys.path.insert(0, 'src')
import ollama
from qdrant_client import QdrantClient

EMBED_MODEL     = "mxbai-embed-large"
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "reglamentos"

cliente = QdrantClient(url=QDRANT_URL)

queries = [
    "retardo hora entrada trabajo sanción",
    "inasistencia injustificada al lugar de trabajo",
    "licencia médica cita especialista permiso",
    "concurrir servicio médico permiso trabajador",
]

for q in queries:
    vector = ollama.embeddings(model=EMBED_MODEL, prompt=q)["embedding"]
    resultados = cliente.search(
        collection_name= COLLECTION_NAME,
        query_vector=    vector,
        limit=           3,
        with_payload=    True
    )
    print(f"\n🔍 '{q}'")
    for r in resultados:
        print(f"   [{r.score:.4f}] p.{r.payload['pagina']} — {r.payload['texto'][:90].strip()}...")