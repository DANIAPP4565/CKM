# App Streamlit CKM + CRHM

Aplicación web para detección, estadificación y manejo clínico-terapéutico del síndrome cardiovascular-renal-metabólico (CKM) y su extensión cardio-reno-hepato-metabólica (CRHM).

## Archivos

- `app_ckm_crhm.py`: código completo integrado de la app.
- `requirements_ckm_crhm.txt`: dependencias mínimas.

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_ckm_crhm.txt
streamlit run app_ckm_crhm.py
```

## Despliegue en Streamlit Cloud

1. Crear un repositorio en GitHub.
2. Subir `app_ckm_crhm.py` y `requirements_ckm_crhm.txt`.
3. En Streamlit Cloud, seleccionar el repositorio.
4. Main file path: `app_ckm_crhm.py`.
5. Requirements file: renombrar `requirements_ckm_crhm.txt` a `requirements.txt` o copiar su contenido.

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
