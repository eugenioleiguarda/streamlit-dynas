# Estado actual reproducible

Inventario generado el 29 de julio de 2026 después de incorporar el análisis
temporal de bloqueo.

## Versión

```text
PIPELINE_CACHE_VERSION = 2026-07-29-tendencias-bloqueo-v16
```

## Archivos ejecutables y SHA-256

| Archivo | SHA-256 |
|---|---|
| `pipeline_diagnostico.py` | `6608c7db8a4a0d4d66b10cfe439a4e93c89f958b0820ad97a4244620c9f66d82` |
| `app_pozos.py` | `c996fa0013b1168ba861ceab5f0d7d21cd81cd7d9ab5d9b0ceceff483f7ee77e` |
| `vfm_produccion.py` | `a9c5323b740aa7362b2a20d7d5cb7fe39ec8e70d90b5e9cc85dbbe856bb4328d` |
| `controles_reales.py` | `8c4863d621bfe958bb2df764186cf50d5993adb17d64864fcb149232602762a6` |
| `requirements.txt` | `52aa739ea9517d5a548e3b7634c94e2a199c0b4b854a9e4a6e0c046462bf38c2` |

Estos hashes permiten comprobar si un archivo local o desplegado coincide con
el estado documentado. En PowerShell:

```powershell
Get-FileHash .\pipeline_diagnostico.py -Algorithm SHA256
Get-FileHash .\app_pozos.py -Algorithm SHA256
```

## Validaciones realizadas

- compilación correcta de:
  - `pipeline_diagnostico.py`;
  - `app_pozos.py`;
  - `vfm_produccion.py`;
  - `controles_reales.py`;
- enlaces internos de la documentación verificados;
- documentación guardada en UTF-8 sin caracteres de reemplazo;
- análisis temporal de bloqueo probado sintéticamente con:
  - bloqueo persistente;
  - bloqueo reciente;
  - bloqueo intermitente;
  - desbloqueo reciente.

## Estado funcional

- Aplicación principal: `app_pozos.py`.
- Diagnóstico por carta: operativo.
- Consolidación robusta 3 de 5: operativa.
- VFM: integrado.
- Comparación con controles: integrada.
- Tendencias históricas: integradas.
- Análisis temporal:
  - subexplotación: integrado;
  - falta de aporte: integrado;
  - bloqueo/sin trabajo: integrado.

## Nota de repositorio

La carpeta local documentada no está inicializada como repositorio Git. Para
conservar este estado en GitHub, subir todos los archivos modificados y crear
un commit/tag. El hash del commit debe agregarse aquí cuando exista.

