# App Streamlit CKM + CRHM

Aplicación web para detección, estadificación y manejo clínico-terapéutico del síndrome cardiovascular-renal-metabólico (CKM) y su extensión cardio-reno-hepato-metabólica (CRHM).

## Archivos

- `app_ckm_crhm.py`: código completo integrado de la app.
- `requirements.txt`: dependencias mínimas para Streamlit Cloud.

Esta versión no utiliza Plotly para evitar errores `ModuleNotFoundError` en despliegues de Streamlit Cloud. Los gráficos se generan con componentes nativos de Streamlit.

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_ckm_crhm.py
```

## Despliegue en Streamlit Cloud

1. Crear un repositorio en GitHub.
2. Subir `app_ckm_crhm.py` y `requirements.txt`.
3. En Streamlit Cloud, seleccionar el repositorio.
4. Main file path: `app_ckm_crhm.py`.
5. Verificar que el archivo de dependencias se llame exactamente `requirements.txt`.

## Módulos incluidos

- Etapa CKM 0-4.
- Etapa CRHM 0-IVb.
- IMC, cintura y adiposidad disfuncional.
- Síndrome metabólico 0-5 criterios.
- Riesgo renal KDIGO operativo con eGFR/UACR.
- FIB-4 y ruta MASLD: seguimiento, elastografía o hepatología.
- PREVENT manual 10/30 años.
- Recomendaciones terapéuticas integradas por dominio.
- Exportación individual a Markdown, Excel y JSON.
- Carga masiva CSV con plantilla.

## Nota clínica

Herramienta de apoyo operativo y educativo. No reemplaza juicio clínico, confirmación diagnóstica, guías locales ni revisión de indicaciones/contraindicaciones antes de prescribir.
