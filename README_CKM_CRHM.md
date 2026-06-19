# App Streamlit CKM + CRHM v5.0

Aplicación web para detección, estadificación y manejo clínico-terapéutico del síndrome cardiovascular-renal-metabólico (CKM) y cardio-reno-hepato-metabólico (CRHM).

## Funciones principales

- Importación de PDF digital de laboratorio con extracción de glucemia, HbA1c, lípidos, eGFR/UACR, AST/ALT, plaquetas, ApoB, no-HDL-C y potasio.
- Estadificación CKM 0-4 y CRHM 0-IV.
- FIB-4 y orientación de elastografía/derivación hepatológica.
- Riesgo renal KDIGO por eGFR y UACR.
- PREVENT: ingreso de resultados oficiales PREVENT-CVD, PREVENT-ASCVD y PREVENT-HF a 10 y 30 años.
- Punto de extensión para coeficientes PREVENT validados mediante archivo local `prevent_coefficients.json`.
- Objetivos LDL-C, no-HDL-C y ApoB por perfil de riesgo.
- Módulo farmacológico con dosis orientativas, contraindicaciones, interacciones y alertas por eGFR/potasio.
- Gráficas interactivas con Altair.
- Login local, roles de médico/admin e historial por usuario.
- Informe PDF médico institucional con logo, firma/sello digital, tablas, mapa de dominios, plan terapéutico y alertas.
- Exportación individual a PDF, Markdown, Excel y JSON.
- Evaluación masiva por CSV.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app_ckm_crhm.py
```

## Streamlit Cloud

Estructura recomendada del repositorio:

```text
ckm/
├── app_ckm_crhm.py
├── requirements.txt
└── README_CKM_CRHM.md
```

En Streamlit Cloud, configurar como archivo principal:

```text
app_ckm_crhm.py
```

Luego ejecutar:

```text
Manage app -> Reboot app
```

Si toma una versión anterior:

```text
Manage app -> Settings -> Clear cache -> Reboot
```

## Login inicial

La app crea un usuario local inicial si no existe base de usuarios:

```text
Usuario: admin
Contraseña: admin123
```

Cambiar esta contraseña en producción. El almacenamiento local en Streamlit Cloud puede reiniciarse si se reconstruye la app; para uso institucional persistente se recomienda conectar una base externa.

## PREVENT

La app no inventa coeficientes PREVENT. Permite ingresar resultados oficiales calculados fuera de la app y deja el código preparado para incorporar coeficientes validados/auditados si se dispone de ellos.

## Seguridad clínica

Herramienta de apoyo a la decisión clínica. No reemplaza juicio clínico, confirmación diagnóstica ni prescripción individualizada. Confirmar indicaciones, contraindicaciones, interacciones, función renal, potasio, estado hepático, embarazo/lactancia y cobertura antes de indicar tratamientos.

## Autor

Dr. Olano Ricardo Daniel, Cardiólogo Hipertensólogo.
