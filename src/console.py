"""
console.py
----------
Helpers pequeños para que los scripts de consola funcionen bien en Windows.
"""

import sys


def configurar_salida_utf8() -> None:
    """
    Fuerza stdout/stderr a UTF-8 cuando el stream lo permite.

    Evita errores UnicodeEncodeError en Windows al imprimir tildes, flechas
    o iconos en terminales configuradas como cp1252.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
