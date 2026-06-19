# App Streamlit CKM + CRHM

App web para detección, estadificación y manejo clínico-terapéutico del síndrome cardiovascular-renal-metabólico (CKM) y cardio-reno-hepato-metabólico (CRHM).

**Autor/Desarrollador:** Dr. Olano Ricardo Daniel, Cardiólogo Hipertensólogo.

## Funciones principales

- Ingreso clínico individual completo.
- Importación de PDF digital de laboratorio.
- Prellenado automático de variables clave:
  - Glucemia en ayunas.
  - HbA1c.
  - Triglicéridos.
  - HDL-C.
  - LDL-C.
  - eGFR/TFG.
  - UACR/CACU.
  - AST/TGO.
  - ALT/TGP.
  - Plaquetas.
  - Creatinina, con cálculo eGFR CKD-EPI 2021 cuando no viene informado el filtrado.
- Estadificación CKM 0-4.
- Extensión CRHM con dominio hepático/MASLD y FIB-4.
- Evaluación KDIGO por eGFR/UACR.
- Gráficas interactivas con Altair.
- Informe PDF médico profesional y didáctico.
- Exportación individual en PDF, Markdown, Excel y JSON.
- Carga masiva por CSV.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app_ckm_crhm.py
```

## Estructura recomendada del repositorio

```text
ckm/
├── app_ckm_crhm.py
├── requirements.txt
└── README_CKM_CRHM.md
```

## Importante sobre PDF de laboratorio

La importación funciona mejor con PDFs digitales, es decir, aquellos donde el texto se puede seleccionar/copiar. Si el PDF es una imagen escaneada, la app mostrará aviso y será necesario realizar OCR previo o ingresar los datos manualmente.

La app no guarda nombres ni DNI. Cuando importa un laboratorio, genera un código anónimo tipo `LAB-XXXXXXXX` a partir del texto del PDF.

## Dependencias

- streamlit
- pandas
- numpy
- openpyxl
- altair
- reportlab
- pypdf

## Nota médica

La app es una herramienta de apoyo clínico y educación médica. No reemplaza el juicio clínico, las guías locales, la confirmación diagnóstica ni la prescripción individualizada.
