import sys
sys.path.insert(0, 'src')
from retriever import buscar_chunks_relevantes

temas = [
    "servicio médico correspondiente",
    "citas médicas especialistas",
    "retardo hora entrada",
    "falta trabajo sin excusa",
    "permiso calamidad doméstica",
    "licencia remunerada",
    "suspensión disciplinaria",
]

for tema in temas:
    chunks = buscar_chunks_relevantes(tema)
    if chunks:
        print(f"\n✅ [{chunks[0]['score']}] '{tema}'")
        print(f"   p.{chunks[0]['pagina']} — {chunks[0]['texto'][:120].strip()}...")
    else:
        print(f"\n❌ Sin resultados para: '{tema}'")