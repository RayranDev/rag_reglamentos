import sys
sys.path.insert(0, 'src')
from retriever import buscar_chunks_relevantes

preguntas = [
    "Últimamente he tenido problemas con el transporte y he llegado tarde varias veces",
    "Tengo una cita médica y no sé si eso toca pedirlo antes",
]

for p in preguntas:
    print(f"\n🔍 {p[:60]}")
    chunks = buscar_chunks_relevantes(p)
    if not chunks:
        print("  ⚠️  Sin resultados con el nuevo threshold")
    for c in chunks:
        print(f"  [{c['score']}] p.{c['pagina']} — {c['texto'][:80].strip()}...")