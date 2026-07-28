"""
Punto de entrada principal de Streamlit.

La URL histórica continúa ejecutando app.py, pero la interfaz vigente
se mantiene en app_pozos.py. Se usa run_path porque Streamlit vuelve a
ejecutar este archivo ante cada interacción; una importación normal
quedaría almacenada en Python y produciría una pantalla vacía.
"""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("app_pozos.py")),
    run_name="__main__",
)
