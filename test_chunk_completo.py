import sys
sys.path.insert(0, 'src')
import ollama
from qdrant_client import QdrantClient

cliente = QdrantClient(url="http://localhost:6333")
vector = ollama.embeddings(model="mxbai-embed-large", prompt="retardo hora entrada trabajo sanción")["embedding"]

resultados = cliente.search(
    collection_name="reglamentos",
    query_vector=vector,
    limit=6,
    with_payload=True
)

for r in resultados:
    print(f"\n[{r.score:.4f}] p.{r.payload['pagina']}")
    print(r.payload['texto'])
    print("─" * 60)