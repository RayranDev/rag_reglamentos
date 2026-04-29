import sys
sys.path.insert(0, 'src')
from retriever import buscar_chunks_relevantes, formatear_contexto

preguntas = [
    "llegué tarde varias veces, ¿me pueden sancionar?",
    "tengo una cita médica, ¿cómo pido el permiso?",
]

for p in preguntas:
    print(f"\n{'='*60}")
    print(f"PREGUNTA: {p}")
    print('='*60)
    chunks = buscar_chunks_relevantes(p)
    for c in chunks:
        print(f"\n[{c['score']}] p.{c['pagina']}")
        print(c['texto'][:200])
        print("─"*40)