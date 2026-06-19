# App CKM + CRHM Streamlit

Aplicación web para detección, estadificación y manejo clínico-terapéutico integrado del síndrome cardiovascular-renal-metabólico (CKM) y cardio-reno-hepato-metabólico (CRHM).

## Versión
v2.0 - 2026 | interactiva sin Plotly | CSV corregido

## Correcciones incluidas

- Eliminada la dependencia directa de Plotly para evitar `ModuleNotFoundError`.
- Agregadas gráficas interactivas con Altair, con fallback seguro a gráficos nativos de Streamlit.
- Corregido el error de plantilla CSV por asignación `df.loc` incompatible con pandas/pyarrow en Streamlit Cloud.
- El título de la app fue bajado visualmente y rediseñado para mejorar legibilidad.
- Se agregaron columnas en la plantilla CSV para aterosclerosis subclínica y apnea del sueño.
- Exportación individual a Markdown, Excel y JSON.
- Carga masiva CSV con exportación a Excel.

## Archivos necesarios

```text
ckm/
├── app_ckm_crhm.py
├── requirements.txt
└── README_CKM_CRHM.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app_ckm_crhm.py
```

## Deploy en Streamlit Cloud

1. Subir `app_ckm_crhm.py`, `requirements.txt` y `README_CKM_CRHM.md` al repositorio.
2. Verificar que el archivo se llame exactamente `requirements.txt`.
3. En Streamlit Cloud seleccionar como archivo principal: `app_ckm_crhm.py`.
4. Reiniciar la app desde `Manage app -> Reboot app`.
5. Si persiste una versión anterior: `Manage app -> Settings -> Clear cache -> Reboot`.

## Nota clínica

La app es una herramienta operativa y educativa. No reemplaza el juicio clínico, la confirmación diagnóstica ni la prescripción individualizada.
