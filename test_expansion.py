import sys
sys.path.insert(0, 'src')
from retriever import expandir_query

preguntas = [
    "llegué tarde varias veces, ¿me pueden sancionar?",
    "tengo una cita médica, ¿cómo pido el permiso?",
    "¿qué pasa si falto sin avisar?",
]

for p in preguntas:
    print(f"\n📝 Original: {p}")
    variantes = expandir_query(p)
    for i, v in enumerate(variantes):
        print(f"   {'Original' if i==0 else f'Variante {i}'}: {v}")