import sys
import unicodedata

sys.path.insert(0, "src")

from console import configurar_salida_utf8
from retriever import buscar_chunks_relevantes


configurar_salida_utf8()


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFD", texto.lower()) \
        .encode("ascii", "ignore").decode("ascii")


CASOS = [
    {
        "pregunta": "retardo hora entrada",
        "debe_contener": ["articulo 88", "retardo"],
    },
    {
        "pregunta": "falta trabajo sin excusa",
        "debe_contener": ["articulo 88", "sin excusa"],
    },
    {
        "pregunta": "tengo una cita medica, como pido el permiso",
        "debe_contener": ["citas medicas", "constancia de agendamiento"],
    },
    {
        "pregunta": "permiso calamidad domestica",
        "debe_contener": ["calamidad domestica", "articulo 45"],
    },
]


def main() -> None:
    for caso in CASOS:
        chunks = buscar_chunks_relevantes(caso["pregunta"])
        assert chunks, f"Sin resultados para: {caso['pregunta']}"

        texto_top = normalizar("\n".join(c["texto"] for c in chunks[:3]))
        esperado = [normalizar(t) for t in caso["debe_contener"]]
        assert any(t in texto_top for t in esperado), (
            f"No se encontro evidencia esperada para: {caso['pregunta']}. "
            f"Esperado alguno de: {caso['debe_contener']}"
        )

        print(f"OK: {caso['pregunta']} -> {chunks[0]['fuente']} p.{chunks[0]['pagina']}")


if __name__ == "__main__":
    main()
