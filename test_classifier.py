import sys
sys.path.insert(0, 'src')
from classifier import clasificar_pregunta

preguntas = [
    "¿Qué pasa si me siento enfermo y no puedo ir a trabajar?",
    "¿Qué debo hacer si llego tarde al trabajo?",
    "Si llego tarde varias veces en el mes, ¿puede haber sanción?",
    "¿Qué pasa si falto al trabajo sin avisar?",
]

for p in preguntas:
    r = clasificar_pregunta(p)
    print(f'[{r["resultado"]}] capa:{r["capa_usada"]} score:{r["score"]} — {p[:55]}')