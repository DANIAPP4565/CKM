# -*- coding: utf-8 -*-
"""
App Streamlit: Síndrome Cardiovascular-Renal-Metabólico (CKM)
y Cardio-Reno-Hepato-Metabólico (CRHM)
Autor: Dr. Olano Ricardo Daniel, Cardiólogo Hipertensólogo

Uso:
    streamlit run app_ckm_crhm.py

Notas clínicas:
- Esta herramienta es un apoyo educativo/operativo para estratificación y manejo.
- No reemplaza el juicio clínico, la confirmación diagnóstica ni la prescripción individualizada.
- PREVENT se ingresa manualmente porque no se implementan coeficientes oficiales cerrados.
"""

from __future__ import annotations

import io
import json
import math
import re
import hashlib
from html import escape as html_escape
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Lectura de PDFs de laboratorio: pypdf se usa de forma opcional.
# Si el PDF es escaneado como imagen, se mostrará aviso porque no se hace OCR.
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except Exception:
    PdfReader = None
    PYPDF_AVAILABLE = False

# Gráficas interactivas: Altair se usa de forma opcional.
# Si Streamlit Cloud no instala la dependencia, la app NO se cae y usa gráficos nativos.
try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except Exception:
    alt = None
    ALTAIR_AVAILABLE = False

# Importación defensiva: no se usa Plotly en esta versión para evitar ModuleNotFoundError.
PLOTLY_AVAILABLE = False

# PDF profesional: ReportLab se usa de forma opcional pero se incluye en requirements.txt.
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

import streamlit as st


APP_TITLE = "Síndrome CKM + CRHM"
APP_SUBTITLE = "Detección, estadificación y manejo clínico-terapéutico integrado"
AUTHOR = "Dr. Olano Ricardo Daniel, Cardiólogo Hipertensólogo"
VERSION = "v4.0 - 2026 | PDF laboratorio + informe médico profesional"


# -----------------------------------------------------------------------------
# Configuración visual
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CKM + CRHM",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
  --primary: #0f766e;
  --primary2: #0ea5e9;
  --danger: #b91c1c;
  --warn: #b45309;
  --ok: #15803d;
  --soft: #f8fafc;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
}
.block-container {padding-top: 10.0rem; padding-bottom: 2rem;}
.app-hero {
  margin-top: 4.2rem; margin-bottom: 1.25rem;
  background: linear-gradient(135deg, #ecfeff 0%, #f8fafc 48%, #eef2ff 100%);
  border: 1px solid var(--border); border-radius: 22px;
  padding: 1.35rem 1.45rem;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
.main-title {
  font-size: clamp(1.75rem, 3vw, 2.65rem); line-height: 1.15; font-weight: 950; letter-spacing: -0.03em;
  color: #020617; margin-bottom: 0.65rem; overflow-wrap: anywhere; text-wrap: balance;
}
.subtitle {color: #334155; font-size: 1.02rem; line-height: 1.45; margin-bottom: 0;}
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.05rem 1.15rem; box-shadow: 0 8px 25px rgba(15, 23, 42, 0.04);
  margin-bottom: 0.85rem;
}
.card h3 {margin: 0 0 0.45rem 0; font-size: 1.05rem;}
.small {font-size: 0.88rem; color: var(--muted);}
.badge {display:inline-block; padding:0.25rem 0.6rem; border-radius:999px; font-weight:700; font-size:0.84rem;}
.badge-ok {background:#dcfce7; color:#166534;}
.badge-warn {background:#fef3c7; color:#92400e;}
.badge-danger {background:#fee2e2; color:#991b1b;}
.badge-info {background:#e0f2fe; color:#075985;}
.kpi {font-size: 1.75rem; font-weight: 850; letter-spacing:-0.02em;}
.kpi-label {font-size: 0.86rem; color: var(--muted);}
.hr {border-top:1px solid var(--border); margin:0.8rem 0;}
.reco-high {border-left: 6px solid #b91c1c;}
.reco-mid {border-left: 6px solid #b45309;}
.reco-low {border-left: 6px solid #15803d;}
.footer {font-size:0.82rem; color:#64748b; margin-top:2rem;}
.chart-note {font-size:0.85rem; color:#64748b; margin-top:0.25rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Utilidades clínicas
# -----------------------------------------------------------------------------
def is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False


def clean_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None or x == "":
        return default
    try:
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt(x: Any, digits: int = 1, na: str = "No disponible") -> str:
    if not is_number(x):
        return na
    return f"{float(x):.{digits}f}"


def safe_file_part(value: Any) -> str:
    text = str(value or "Paciente").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned[:60] or "Paciente"


def bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if not is_number(weight_kg) or not is_number(height_cm) or height_cm <= 0:
        return None
    return float(weight_kg) / (float(height_cm) / 100) ** 2


def bmi_class(b: Optional[float]) -> str:
    if b is None:
        return "No disponible"
    if b < 18.5:
        return "Bajo peso"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Sobrepeso"
    if b < 35:
        return "Obesidad clase I"
    if b < 40:
        return "Obesidad clase II"
    return "Obesidad clase III"


def waist_threshold(sex: str, asian_ancestry: bool = False) -> float:
    if sex == "Mujer":
        return 80.0 if asian_ancestry else 88.0
    return 90.0 if asian_ancestry else 102.0


def abnormal_waist(sex: str, waist_cm: Optional[float], asian_ancestry: bool = False) -> bool:
    if not is_number(waist_cm):
        return False
    return float(waist_cm) >= waist_threshold(sex, asian_ancestry)


def hdl_low(sex: str, hdl: Optional[float]) -> bool:
    if not is_number(hdl):
        return False
    return float(hdl) < (50.0 if sex == "Mujer" else 40.0)


def elevated_bp(sbp: Optional[float], dbp: Optional[float], on_bp_meds: bool) -> bool:
    sbp_high = is_number(sbp) and float(sbp) >= 130
    dbp_high = is_number(dbp) and float(dbp) >= 80
    return bool(sbp_high or dbp_high or on_bp_meds)


def prediabetes(fpg: Optional[float], hba1c: Optional[float], ogtt_2h: Optional[float]) -> bool:
    fpg_pre = is_number(fpg) and 100 <= float(fpg) <= 125
    a1c_pre = is_number(hba1c) and 5.7 <= float(hba1c) <= 6.4
    ogtt_pre = is_number(ogtt_2h) and 140 <= float(ogtt_2h) <= 199
    return bool(fpg_pre or a1c_pre or ogtt_pre)


def diabetes_by_labs(fpg: Optional[float], hba1c: Optional[float], ogtt_2h: Optional[float]) -> bool:
    fpg_dm = is_number(fpg) and float(fpg) >= 126
    a1c_dm = is_number(hba1c) and float(hba1c) >= 6.5
    ogtt_dm = is_number(ogtt_2h) and float(ogtt_2h) >= 200
    return bool(fpg_dm or a1c_dm or ogtt_dm)


def metabolic_syndrome_count(
    sex: str,
    waist_cm: Optional[float],
    tg: Optional[float],
    hdl: Optional[float],
    sbp: Optional[float],
    dbp: Optional[float],
    on_bp_meds: bool,
    fpg: Optional[float],
    known_dm: bool,
    asian_ancestry: bool = False,
) -> Tuple[int, Dict[str, bool]]:
    criteria = {
        "Cintura aumentada": abnormal_waist(sex, waist_cm, asian_ancestry),
        "HDL bajo": hdl_low(sex, hdl),
        "Triglicéridos ≥150": is_number(tg) and float(tg) >= 150,
        "PA elevada/tratada": elevated_bp(sbp, dbp, on_bp_meds),
        "Glucosa ≥100 o diabetes": (is_number(fpg) and float(fpg) >= 100) or known_dm,
    }
    return sum(criteria.values()), criteria


def egfr_g_category(egfr: Optional[float]) -> str:
    if not is_number(egfr):
        return "G?"
    e = float(egfr)
    if e >= 90:
        return "G1"
    if e >= 60:
        return "G2"
    if e >= 45:
        return "G3a"
    if e >= 30:
        return "G3b"
    if e >= 15:
        return "G4"
    return "G5"


def albumin_a_category(uacr: Optional[float]) -> str:
    if not is_number(uacr):
        return "A?"
    a = float(uacr)
    if a < 30:
        return "A1"
    if a < 300:
        return "A2"
    return "A3"


def kdigo_risk(egfr: Optional[float], uacr: Optional[float]) -> Tuple[str, str, str]:
    """Clasificación simplificada, alineada con el uso operativo CKM."""
    g = egfr_g_category(egfr)
    a = albumin_a_category(uacr)
    if g == "G?" or a == "A?":
        return g, a, "No disponible"
    very_high = (
        (g == "G3a" and a == "A3")
        or (g == "G3b" and a in ["A2", "A3"])
        or (g in ["G4", "G5"])
    )
    moderate_high = (
        (g in ["G1", "G2"] and a in ["A2", "A3"])
        or (g == "G3a" and a in ["A1", "A2"])
        or (g == "G3b" and a == "A1")
    )
    if very_high:
        return g, a, "Muy alto"
    if moderate_high:
        return g, a, "Moderado-alto"
    return g, a, "Bajo"


def fib4(age: Optional[float], ast: Optional[float], alt: Optional[float], platelets: Optional[float]) -> Optional[float]:
    if not all(is_number(v) for v in [age, ast, alt, platelets]):
        return None
    if float(alt) <= 0 or float(platelets) <= 0:
        return None
    return (float(age) * float(ast)) / (float(platelets) * math.sqrt(float(alt)))


def fib4_category(age: Optional[float], fib4_value: Optional[float]) -> Tuple[str, str]:
    if not is_number(age) or fib4_value is None:
        return "No disponible", "Completar edad, AST, ALT y plaquetas."
    a = float(age)
    f = float(fib4_value)
    if a < 35:
        return "No aplicable <35 años", "FIB-4 tiene menor utilidad en menores de 35 años; considerar elastografía si hay alta sospecha."
    if a < 65:
        if f < 1.3:
            return "Bajo", "Seguimiento rutinario y control de factores metabólicos."
        if f <= 2.67:
            return "Intermedio", "Solicitar elastografía hepática (VCTE/FibroScan) o ELF."
        return "Alto", "Derivar a hepatología / evaluar fibrosis avanzada."
    # >=65 años
    if f < 2.0:
        return "Bajo", "Seguimiento rutinario y control de factores metabólicos."
    if f <= 2.67:
        return "Intermedio", "Solicitar elastografía hepática (VCTE/FibroScan) o ELF."
    return "Alto", "Derivar a hepatología / evaluar fibrosis avanzada."


def infer_ldl_priority(stage: str, has_ascvd: bool, known_dm: bool, kdigo: str, prevent_10: Optional[float], ldl: Optional[float]) -> str:
    if has_ascvd:
        return "Muy alta prioridad: prevención secundaria. Intensificar estatina/ezetimibe/PCSK9 según LDL y riesgo."
    if is_number(ldl) and float(ldl) >= 190:
        return "Muy alta prioridad: LDL ≥190 mg/dL; descartar hipercolesterolemia familiar."
    if known_dm or kdigo in ["Moderado-alto", "Muy alto"]:
        return "Alta prioridad: diabetes/ERC elevan riesgo; indicar tratamiento lipídico según guías locales."
    if is_number(prevent_10) and float(prevent_10) >= 7.5:
        return "Prioridad aumentada: PREVENT ≥7,5%; discutir estatina y objetivos de LDL."
    return "Prioridad según riesgo global, edad, LDL y decisión compartida."


def semaforo_level(stage_numeric: int, crhm_stage: str, fib4_cat: str, kdigo: str) -> Tuple[str, str]:
    if stage_numeric >= 4 or crhm_stage.startswith("IV") or fib4_cat == "Alto" or kdigo == "Muy alto":
        return "Rojo", "badge-danger"
    if stage_numeric == 3 or crhm_stage == "III" or fib4_cat == "Intermedio" or kdigo == "Moderado-alto":
        return "Naranja", "badge-warn"
    if stage_numeric in [1, 2] or crhm_stage in ["I", "II"]:
        return "Amarillo", "badge-warn"
    return "Verde", "badge-ok"


# -----------------------------------------------------------------------------
# Evaluación integral
# -----------------------------------------------------------------------------
def evaluate_patient(p: Dict[str, Any]) -> Dict[str, Any]:
    age = clean_float(p.get("age"))
    sex = p.get("sex", "Hombre")
    asian = bool(p.get("asian_ancestry", False))
    weight = clean_float(p.get("weight_kg"))
    height = clean_float(p.get("height_cm"))
    waist = clean_float(p.get("waist_cm"))
    sbp = clean_float(p.get("sbp"))
    dbp = clean_float(p.get("dbp"))
    on_bp_meds = bool(p.get("on_bp_meds", False))
    fpg = clean_float(p.get("fpg"))
    ogtt = clean_float(p.get("ogtt_2h"))
    hba1c = clean_float(p.get("hba1c"))
    tg = clean_float(p.get("tg"))
    hdl = clean_float(p.get("hdl"))
    ldl = clean_float(p.get("ldl"))
    egfr = clean_float(p.get("egfr"))
    uacr = clean_float(p.get("uacr"))
    ast = clean_float(p.get("ast"))
    alt = clean_float(p.get("alt"))
    platelets = clean_float(p.get("platelets"))
    prevent_10 = clean_float(p.get("prevent_10"))
    prevent_30 = clean_float(p.get("prevent_30"))

    b = bmi(weight, height)
    adiposity = bool((is_number(b) and b >= (23 if asian else 25)) or abnormal_waist(sex, waist, asian))
    obesity = bool(is_number(b) and b >= 30)
    severe_obesity = bool(is_number(b) and b >= 40)
    pre_dm = prediabetes(fpg, hba1c, ogtt)
    lab_dm = diabetes_by_labs(fpg, hba1c, ogtt)
    known_t2d = bool(p.get("known_t2d", False)) or lab_dm
    htn = elevated_bp(sbp, dbp, on_bp_meds)
    ms_count, ms_criteria = metabolic_syndrome_count(sex, waist, tg, hdl, sbp, dbp, on_bp_meds, fpg, known_t2d, asian)
    metabolic_syndrome = ms_count >= 3
    hypertriglyceridemia = is_number(tg) and float(tg) >= 150
    metabolic_rf = bool(hypertriglyceridemia or htn or metabolic_syndrome or known_t2d)

    gcat, acat, k_risk = kdigo_risk(egfr, uacr)
    ckd_any = bool((is_number(egfr) and float(egfr) < 60) or (is_number(uacr) and float(uacr) >= 30))
    kdigo_moderate_high = k_risk == "Moderado-alto"
    kdigo_very_high = k_risk == "Muy alto"
    esrd_or_dialysis = bool((is_number(egfr) and float(egfr) < 15) or p.get("dialysis", False) or p.get("kidney_transplant", False))

    asc = bool(p.get("ascvd", False))
    hf = bool(p.get("hf", False))
    stroke_tia = bool(p.get("stroke_tia", False))
    pad = bool(p.get("pad", False))
    af = bool(p.get("af", False))
    clinical_cvd = asc or hf or stroke_tia or pad or af

    cac_gt100 = bool(p.get("cac_gt100", False))
    pre_hf = bool(p.get("pre_hf", False))
    subclinical_cvd = bool(cac_gt100 or pre_hf or p.get("subclinical_atherosclerosis", False))

    masld_known = bool(p.get("masld_known", False))
    liver_stage = p.get("liver_stage", "Desconocido")
    fib4_val = fib4(age, ast, alt, platelets)
    fib4_cat, fib4_action = fib4_category(age, fib4_val)
    significant_fibrosis = bool(liver_stage in ["F2-F4", "Cirrhosis"] or fib4_cat == "Alto")
    cirrhosis = liver_stage == "Cirrhosis"

    ckm_rf_any = adiposity or pre_dm or metabolic_rf or ckd_any

    # CKM AHA/ACC/ADA/ASN operativo
    if clinical_cvd and ckm_rf_any:
        ckm_stage = "Etapa 4: ECV clínica con factores CKM"
        ckm_stage_n = 4
    elif (subclinical_cvd and ckm_rf_any) or kdigo_very_high or (is_number(prevent_10) and float(prevent_10) >= 20):
        ckm_stage = "Etapa 3: ECV subclínica o equivalente de muy alto riesgo"
        ckm_stage_n = 3
    elif metabolic_rf or kdigo_moderate_high or ckd_any:
        ckm_stage = "Etapa 2: factores metabólicos, ERC o ambos"
        ckm_stage_n = 2
    elif adiposity or pre_dm:
        ckm_stage = "Etapa 1: adiposidad excesiva/disfuncional"
        ckm_stage_n = 1
    else:
        ckm_stage = "Etapa 0: sin factores CKM detectados"
        ckm_stage_n = 0

    # CRHM: extensión cardio-reno-hepato-metabólica operativa
    if clinical_cvd and (ckm_rf_any or masld_known):
        if esrd_or_dialysis or cirrhosis:
            crhm_stage = "IVb"
            crhm_label = "CRHM IVb: ECV clínica con ERC terminal y/o cirrosis"
        else:
            crhm_stage = "IVa"
            crhm_label = "CRHM IVa: ECV clínica sin ERC terminal/cirrosis"
    elif subclinical_cvd or kdigo_very_high or significant_fibrosis or (is_number(prevent_10) and float(prevent_10) >= 20):
        crhm_stage = "III"
        crhm_label = "CRHM III: enfermedad subclínica o equivalente de alto riesgo"
    elif metabolic_rf or ckd_any or masld_known:
        crhm_stage = "II"
        crhm_label = "CRHM II: factores metabólicos, renales o hepáticos"
    elif adiposity or pre_dm:
        crhm_stage = "I"
        crhm_label = "CRHM I: adiposidad excesiva/disfuncional"
    else:
        crhm_stage = "0"
        crhm_label = "CRHM 0: sin factores detectados"

    sem, sem_cls = semaforo_level(ckm_stage_n, crhm_stage, fib4_cat, k_risk)

    domain_scores = {
        "Adiposidad": 3 if severe_obesity else 2 if obesity else 1 if adiposity else 0,
        "Metabólico": 3 if known_t2d else 2 if metabolic_syndrome else 1 if (pre_dm or hypertriglyceridemia or htn) else 0,
        "Renal": 3 if kdigo_very_high else 2 if kdigo_moderate_high else 1 if ckd_any else 0,
        "Hepático": 3 if (fib4_cat == "Alto" or cirrhosis) else 2 if fib4_cat == "Intermedio" else 1 if masld_known else 0,
        "Cardiovascular": 3 if clinical_cvd else 2 if subclinical_cvd else 1 if (is_number(prevent_10) and float(prevent_10) >= 7.5) else 0,
    }

    return {
        "patient_code": p.get("patient_code", "Paciente"),
        "age": age,
        "sex": sex,
        "asian_ancestry": asian,
        "bmi": b,
        "bmi_class": bmi_class(b),
        "waist_threshold": waist_threshold(sex, asian),
        "abnormal_waist": abnormal_waist(sex, waist, asian),
        "prediabetes": pre_dm,
        "t2d": known_t2d,
        "hypertension": htn,
        "hypertriglyceridemia": hypertriglyceridemia,
        "hdl_low": hdl_low(sex, hdl),
        "metabolic_syndrome_count": ms_count,
        "metabolic_syndrome": metabolic_syndrome,
        "metabolic_syndrome_criteria": ms_criteria,
        "egfr_category": gcat,
        "albuminuria_category": acat,
        "kdigo_risk": k_risk,
        "ckd_any": ckd_any,
        "fib4": fib4_val,
        "fib4_category": fib4_cat,
        "fib4_action": fib4_action,
        "masld_known": masld_known,
        "liver_stage": liver_stage,
        "significant_fibrosis": significant_fibrosis,
        "clinical_cvd": clinical_cvd,
        "subclinical_cvd": subclinical_cvd,
        "ascvd": asc,
        "hf": hf,
        "af": af,
        "pad": pad,
        "stroke_tia": stroke_tia,
        "pre_hf": pre_hf,
        "cac_gt100": cac_gt100,
        "prevent_10": prevent_10,
        "prevent_30": prevent_30,
        "ckm_stage": ckm_stage,
        "ckm_stage_n": ckm_stage_n,
        "crhm_stage": crhm_stage,
        "crhm_label": crhm_label,
        "traffic_light": sem,
        "traffic_light_cls": sem_cls,
        "domain_scores": domain_scores,
        "ldl_priority": infer_ldl_priority(ckm_stage, clinical_cvd, known_t2d, k_risk, prevent_10, ldl),
        "raw": p,
    }


# -----------------------------------------------------------------------------
# Recomendaciones de manejo
# -----------------------------------------------------------------------------
def add_reco(recos: List[Dict[str, str]], priority: str, domain: str, recommendation: str, rationale: str, follow_up: str) -> None:
    recos.append({
        "Prioridad": priority,
        "Dominio": domain,
        "Recomendación": recommendation,
        "Fundamento": rationale,
        "Seguimiento": follow_up,
    })


def generate_recommendations(p: Dict[str, Any], r: Dict[str, Any]) -> List[Dict[str, str]]:
    recos: List[Dict[str, str]] = []
    raw = r["raw"]
    bmi_val = r["bmi"]
    has_ckm = r["ckm_stage_n"] > 0

    add_reco(
        recos,
        "Alta" if has_ckm else "Preventiva",
        "Modelo de atención",
        "Manejo integrado CKM/CRHM con coordinación clínica: cardiología, nefrología, endocrinología/metabolismo, nutrición y hepatología si corresponde.",
        "La estratificación por etapas permite ajustar la intensidad terapéutica al riesgo absoluto y evitar atención fragmentada.",
        "Revaluar etapa CKM/CRHM cada 3-6 meses si hay tratamiento activo; anual si bajo riesgo estable.",
    )

    add_reco(
        recos,
        "Alta" if r["ckm_stage_n"] >= 2 else "Preventiva",
        "Estilo de vida",
        "Plan cardiometabólico: patrón mediterráneo/DASH, reducción de ultraprocesados, actividad aeróbica + fuerza, sueño regular, abandono de tabaco y objetivo de peso individualizado.",
        "La intervención sobre estilo de vida y peso es transversal a todas las etapas y puede favorecer regresión de etapa.",
        "Control de peso, cintura, PA domiciliaria y adherencia cada 4-12 semanas al inicio.",
    )

    if is_number(bmi_val) and (bmi_val >= 30 or (bmi_val >= 27 and r["ckm_stage_n"] >= 2)):
        priority = "Alta" if bmi_val >= 30 else "Media"
        add_reco(
            recos,
            priority,
            "Adiposidad/obesidad",
            "Considerar farmacoterapia antiobesidad si no hay contraindicaciones; en perfil CKM/CRHM favorece evaluar terapias basadas en GLP-1 o dual GIP/GLP-1. Considerar cirugía metabólica/bariátrica si IMC muy elevado o comorbilidades relevantes.",
            "El exceso de adiposidad es motor de progresión CKM/CRHM y objetivo terapéutico modificable.",
            "Registrar % de pérdida de peso a 3-6 meses, tolerancia gastrointestinal, masa muscular y riesgo nutricional.",
        )

    if r["hypertension"]:
        add_reco(
            recos,
            "Alta",
            "Hipertensión arterial",
            "Confirmar patrón con MDPA/MAPA si hay duda; intensificar control de PA, sodio, peso y fármacos según fenotipo clínico. En albuminuria/ERC priorizar bloqueo SRAA si tolerado.",
            "La PA elevada es factor CKM y acelera daño cardiovascular, renal y hepático-metabólico.",
            "PA domiciliaria semanal al inicio; creatinina y potasio 1-4 semanas tras iniciar/aumentar IECA/ARA II/MRA.",
        )

    if r["t2d"]:
        if r["clinical_cvd"] or r["ckd_any"] or r["hf"] or r["masld_known"] or r["ckm_stage_n"] >= 2:
            add_reco(
                recos,
                "Alta",
                "Diabetes tipo 2",
                "En DM2 con ECV, ERC, IC, obesidad o MASLD, evaluar SGLT2i y/o terapia basada en GLP-1 como protección cardiometabólica y renal, además del control glucémico individualizado.",
                "Estas clases ofrecen beneficios multiorgánicos; SGLT2i se prioriza especialmente en IC/ERC y GLP-1/GIP-GLP-1 en obesidad/ASCVD/MASLD.",
                "HbA1c cada 3 meses hasta estabilidad; eGFR/UACR y eventos adversos según fármaco.",
            )
        else:
            add_reco(
                recos,
                "Media",
                "Diabetes tipo 2",
                "Optimizar estilo de vida, peso y tratamiento hipoglucemiante con evaluación de riesgo cardiovascular/renal.",
                "La DM2 define mayor riesgo de progresión CKM.",
                "HbA1c y perfil renal cada 3-6 meses.",
            )
    elif r["prediabetes"]:
        add_reco(
            recos,
            "Media",
            "Prediabetes/adiposidad disfuncional",
            "Programa intensivo de pérdida de peso, actividad física y control de cintura; considerar OGTT si discordancia entre glucosa y HbA1c.",
            "La prediabetes refleja adiposidad disfuncional y puede ubicar al paciente en etapa CKM inicial.",
            "Glucemia/HbA1c cada 6-12 meses.",
        )

    if r["ckd_any"] or r["kdigo_risk"] in ["Moderado-alto", "Muy alto"]:
        kidney_reco = "Cuantificar eGFR y UACR seriados. "
        if r["albuminuria_category"] in ["A2", "A3"]:
            kidney_reco += "Si hay albuminuria, priorizar IECA/ARA II si tolerado. "
        kidney_reco += "Evaluar SGLT2i si cumple criterios clínicos y eGFR permite uso. "
        if r["t2d"] and r["albuminuria_category"] in ["A2", "A3"]:
            kidney_reco += "Si albuminuria persiste y potasio/eGFR lo permiten, considerar finerenona."
        add_reco(
            recos,
            "Alta" if r["kdigo_risk"] == "Muy alto" else "Media",
            "Riñón/ERC",
            kidney_reco,
            "La ERC y la albuminuria modifican el riesgo cardiovascular y la selección de terapias protectoras.",
            "eGFR, UACR y potasio cada 3-6 meses; antes si cambios de SRAA/SGLT2i/nsMRA.",
        )

    if r["clinical_cvd"]:
        add_reco(
            recos,
            "Alta",
            "Enfermedad cardiovascular clínica",
            "Prevención secundaria intensiva: control lipídico, PA, antiagregación/anticoagulación cuando corresponda, rehabilitación/actividad física supervisada y terapias CKM cardioprotectoras según comorbilidad.",
            "La presencia de ECV clínica define etapa CKM/CRHM avanzada y mayor beneficio neto de intervenciones intensivas.",
            "Seguimiento estrecho cada 1-3 meses hasta estabilización de objetivos.",
        )

    if r["hf"] or r["pre_hf"]:
        add_reco(
            recos,
            "Alta" if r["hf"] else "Media",
            "Insuficiencia cardíaca / pre-IC",
            "Si ICFEr: optimizar 4 pilares según tolerancia. En ICFEp/ICFEmr, SGLT2i como eje terapéutico; si obesidad, considerar GLP-1/GIP-GLP-1; si DM2/ERC, evaluar finerenona cuando corresponda.",
            "El fenotipo CKM modifica la prevención y el tratamiento de IC.",
            "NT-proBNP/eco según clínica; vigilar volemia, función renal y potasio.",
        )

    if r["af"]:
        add_reco(
            recos,
            "Alta",
            "Fibrilación auricular",
            "Evaluar anticoagulación, control de ritmo/frecuencia y reducción de carga de FA mediante peso, PA, apnea del sueño, diabetes, alcohol y actividad física.",
            "La FA forma parte de la ECV clínica en CKM y comparte determinantes con obesidad, HTA, DM2 y ERC.",
            "Revisar CHA₂DS₂-VASc/HAS-BLED, función renal y adherencia terapéutica.",
        )

    if r["masld_known"] or r["fib4_category"] in ["Intermedio", "Alto"]:
        add_reco(
            recos,
            "Alta" if r["fib4_category"] == "Alto" else "Media",
            "Hígado/MASLD",
            f"FIB-4: {fmt(r['fib4'], 2)} ({r['fib4_category']}). {r['fib4_action']} Manejo base: pérdida de peso, control de DM2/TG/PA, evitar alcohol nocivo y revisar fármacos hepatotóxicos.",
            "El componente hepático amplía el marco CKM hacia CRHM y cambia la necesidad de elastografía/derivación.",
            "Repetir FIB-4 anual si bajo riesgo; elastografía/ELF si intermedio; hepatología si alto o cirrosis.",
        )

    if is_number(raw.get("ldl")) or r["clinical_cvd"] or r["t2d"] or r["ckd_any"]:
        add_reco(
            recos,
            "Alta" if r["clinical_cvd"] or r["t2d"] or r["ckd_any"] else "Media",
            "Lípidos",
            r["ldl_priority"],
            "El riesgo CKM/CRHM ayuda a priorizar intensidad hipolipemiante.",
            "Control de LDL/no-HDL/ApoB a 4-12 semanas tras cambios y luego cada 3-12 meses.",
        )

    if raw.get("osa_suspected", False) or (is_number(bmi_val) and bmi_val >= 30 and r["hypertension"]):
        add_reco(
            recos,
            "Media",
            "Apnea obstructiva del sueño",
            "Aplicar STOP-Bang y solicitar estudio de sueño si riesgo alto, especialmente si HTA resistente, obesidad, FA o IC.",
            "La AOS es frecuente en CKM/CRHM y empeora PA, arritmias, resistencia a insulina e IC.",
            "Revisar síntomas, adherencia a CPAP si indicada y respuesta de PA/FC.",
        )

    return recos


# -----------------------------------------------------------------------------
# Visualizaciones y reportes
# -----------------------------------------------------------------------------
def render_badge(text: str, cls: str = "badge-info") -> str:
    return f"<span class='badge {cls}'>{text}</span>"


def domain_chart(scores: Dict[str, int]) -> pd.DataFrame:
    """DataFrame de carga por dominio. Mantiene compatibilidad con versiones previas."""
    return pd.DataFrame({
        "Dominio": list(scores.keys()),
        "Carga": list(scores.values()),
    }).set_index("Dominio")


def domain_chart_df(scores: Dict[str, int]) -> pd.DataFrame:
    labels = {
        0: "Sin carga evidente",
        1: "Carga leve",
        2: "Carga moderada",
        3: "Carga alta",
    }
    return pd.DataFrame([
        {"Dominio": domain, "Carga": int(score), "Interpretación": labels.get(int(score), "No clasificado")}
        for domain, score in scores.items()
    ])


def render_domain_chart(scores: Dict[str, int]) -> None:
    """Gráfico interactivo de dominios. Usa Altair si está disponible y fallback seguro si no."""
    df = domain_chart_df(scores)
    if ALTAIR_AVAILABLE and alt is not None:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=7)
            .encode(
                x=alt.X("Carga:Q", title="Carga clínica", scale=alt.Scale(domain=[0, 3]), axis=alt.Axis(values=[0, 1, 2, 3])),
                y=alt.Y("Dominio:N", title=None, sort="-x"),
                tooltip=[
                    alt.Tooltip("Dominio:N", title="Dominio"),
                    alt.Tooltip("Carga:Q", title="Carga", format=".0f"),
                    alt.Tooltip("Interpretación:N", title="Interpretación"),
                ],
            )
            .properties(height=285)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("<div class='chart-note'>Gráfico interactivo: pasar el cursor para ver la interpretación por dominio.</div>", unsafe_allow_html=True)
    else:
        st.bar_chart(df.set_index("Dominio")["Carga"])
        st.caption("Carga orientativa por dominio: 0 ausente, 1 leve, 2 moderada, 3 alta.")


def render_stage_chart(stage_n: int) -> None:
    """Visualización interactiva de progresión CKM 0-4."""
    stages = [
        (0, "Etapa 0", "Sin factores CKM"),
        (1, "Etapa 1", "Adiposidad/disfunción adiposa"),
        (2, "Etapa 2", "Factores metabólicos o ERC"),
        (3, "Etapa 3", "ECV subclínica / alto riesgo"),
        (4, "Etapa 4", "ECV clínica"),
    ]
    df = pd.DataFrame([
        {"Etapa": name, "Valor": n, "Activa": "Sí" if n <= stage_n else "No", "Descripción": desc}
        for n, name, desc in stages
    ])
    if ALTAIR_AVAILABLE and alt is not None:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=7)
            .encode(
                x=alt.X("Valor:Q", title="Progresión", scale=alt.Scale(domain=[0, 4]), axis=alt.Axis(values=[0, 1, 2, 3, 4])),
                y=alt.Y("Etapa:N", title=None, sort=["Etapa 4", "Etapa 3", "Etapa 2", "Etapa 1", "Etapa 0"]),
                opacity=alt.condition(alt.datum.Valor <= stage_n, alt.value(0.95), alt.value(0.22)),
                tooltip=["Etapa:N", "Descripción:N", alt.Tooltip("Valor:Q", title="Nivel", format=".0f")],
            )
            .properties(height=230)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.progress(stage_progress_value(stage_n))
        st.caption(f"Etapa {stage_n} de 4")


def render_recommendations_chart(recos: List[Dict[str, Any]]) -> None:
    """Resumen visual interactivo de prioridades terapéuticas."""
    if not recos:
        st.info("Sin recomendaciones automáticas con los datos ingresados.")
        return
    df = pd.DataFrame(recos)
    if "Prioridad" not in df.columns:
        return
    counts = df.groupby("Prioridad", dropna=False).size().reset_index(name="Cantidad")
    order = ["Alta", "Media", "Baja"]
    counts["Orden"] = counts["Prioridad"].apply(lambda x: order.index(x) if x in order else 99)
    counts = counts.sort_values("Orden")
    if ALTAIR_AVAILABLE and alt is not None:
        chart = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusEnd=7)
            .encode(
                x=alt.X("Cantidad:Q", title="Número de acciones"),
                y=alt.Y("Prioridad:N", title=None, sort=order),
                tooltip=["Prioridad:N", alt.Tooltip("Cantidad:Q", format=".0f")],
            )
            .properties(height=180)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(counts.set_index("Prioridad")["Cantidad"])


def stage_progress_value(stage_n: int) -> float:
    """Valor 0-1 para st.progress de etapa CKM."""
    try:
        return min(max(float(stage_n) / 4.0, 0.0), 1.0)
    except Exception:
        return 0.0


def flatten_summary(p: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    raw = r.get("raw", {})
    return {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "codigo_paciente": r.get("patient_code", "Paciente"),
        "edad": r.get("age"),
        "sexo": r.get("sex"),
        "peso_kg": raw.get("weight_kg"),
        "talla_cm": raw.get("height_cm"),
        "imc": r.get("bmi"),
        "clase_imc": r.get("bmi_class"),
        "cintura_cm": raw.get("waist_cm"),
        "cintura_aumentada": r.get("abnormal_waist"),
        "pas": raw.get("sbp"),
        "pad": raw.get("dbp"),
        "hta": r.get("hypertension"),
        "glucemia": raw.get("fpg"),
        "hba1c": raw.get("hba1c"),
        "tg": raw.get("tg"),
        "hdl": raw.get("hdl"),
        "ldl": raw.get("ldl"),
        "prediabetes": r.get("prediabetes"),
        "dm2": r.get("t2d"),
        "sindrome_metabolico_n": r.get("metabolic_syndrome_count"),
        "sindrome_metabolico": r.get("metabolic_syndrome"),
        "egfr": raw.get("egfr"),
        "uacr": raw.get("uacr"),
        "categoria_egfr": r.get("egfr_category"),
        "categoria_albuminuria": r.get("albuminuria_category"),
        "riesgo_kdigo": r.get("kdigo_risk"),
        "ast": raw.get("ast"),
        "alt": raw.get("alt"),
        "plaquetas": raw.get("platelets"),
        "fib4": r.get("fib4"),
        "fib4_categoria": r.get("fib4_category"),
        "masld": r.get("masld_known"),
        "ecv_clinica": r.get("clinical_cvd"),
        "ecv_subclinica": r.get("subclinical_cvd"),
        "prevent_10": r.get("prevent_10"),
        "prevent_30": r.get("prevent_30"),
        "etapa_ckm": r.get("ckm_stage"),
        "etapa_crhm": r.get("crhm_label"),
        "semaforo": r.get("traffic_light"),
    }


def report_markdown(p: Dict[str, Any], r: Dict[str, Any], recos: List[Dict[str, str]]) -> str:
    lines = []
    lines.append(f"# Informe operativo CKM/CRHM")
    lines.append(f"**Autor/Desarrollador:** {AUTHOR}")
    lines.append(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"**Código:** {r.get('patient_code', 'Paciente')}\n")
    lines.append("## Resultado integrado")
    lines.append(f"- **Etapa CKM:** {r['ckm_stage']}")
    lines.append(f"- **Etapa CRHM:** {r['crhm_label']}")
    lines.append(f"- **Semáforo:** {r['traffic_light']}")
    lines.append(f"- **IMC:** {fmt(r['bmi'])} kg/m² ({r['bmi_class']})")
    lines.append(f"- **Síndrome metabólico:** {r['metabolic_syndrome_count']}/5 criterios")
    lines.append(f"- **Riñón:** eGFR {p.get('egfr', 'NA')} ml/min/1,73 m²; UACR {p.get('uacr', 'NA')} mg/g; KDIGO {r['egfr_category']}-{r['albuminuria_category']} ({r['kdigo_risk']})")
    lines.append(f"- **Hígado:** FIB-4 {fmt(r['fib4'], 2)} ({r['fib4_category']}); {r['fib4_action']}")
    lines.append(f"- **Riesgo PREVENT ingresado:** 10 años {fmt(r['prevent_10'])}%; 30 años {fmt(r['prevent_30'])}%\n")
    lines.append("## Recomendaciones")
    for i, rec in enumerate(recos, 1):
        lines.append(f"### {i}. {rec['Dominio']} — prioridad {rec['Prioridad']}")
        lines.append(f"**Recomendación:** {rec['Recomendación']}")
        lines.append(f"**Fundamento:** {rec['Fundamento']}")
        lines.append(f"**Seguimiento:** {rec['Seguimiento']}\n")
    lines.append("---")
    lines.append("Herramienta de apoyo clínico. Confirmar indicaciones, contraindicaciones, dosis e interacciones antes de prescribir.")
    return "\n".join(lines)


def pdf_clean_text(value: Any) -> str:
    """Texto seguro para PDF con fuentes estándar de ReportLab."""
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "≥": ">=", "≤": "<=", "≈": "~", "→": "->", "←": "<-",
        "–": "-", "—": "-", "−": "-", "•": "-", "·": "-",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "²": "2", "³": "3", "⁹": "9", "₁": "1", "₂": "2",
        "🫀": "", "⬇️": "", "✅": "", "⚠️": "", "❌": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pdf_para(text: Any, style: Any) -> Any:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab no está instalado.")
    cleaned = pdf_clean_text(text).replace("\n", "<br/>")
    return Paragraph(html_escape(cleaned), style)


def pdf_rich(text: str, style: Any) -> Any:
    """Permite etiquetas mínimas generadas por la app (<b>, <br/>)."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab no está instalado.")
    cleaned = pdf_clean_text(text).replace("\n", "<br/>")
    return Paragraph(cleaned, style)


def pdf_score_color(score: int):
    if not REPORTLAB_AVAILABLE:
        return None
    if score <= 0:
        return colors.HexColor("#dcfce7")
    if score == 1:
        return colors.HexColor("#dbeafe")
    if score == 2:
        return colors.HexColor("#fef3c7")
    return colors.HexColor("#fee2e2")


def pdf_priority_color(priority: str):
    if not REPORTLAB_AVAILABLE:
        return None
    p = str(priority).lower()
    if "alta" in p:
        return colors.HexColor("#fee2e2")
    if "media" in p:
        return colors.HexColor("#fef3c7")
    if "prevent" in p:
        return colors.HexColor("#e0f2fe")
    return colors.HexColor("#dcfce7")


def pdf_domain_drawing(scores: Dict[str, int]) -> Any:
    """Gráfico horizontal simple de carga por dominios para el PDF."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab no está instalado.")
    width = 500
    row_h = 25
    height = 22 + row_h * len(scores)
    d = Drawing(width, height)
    x_label = 0
    x_bar = 135
    bar_w = 285
    y = height - 24
    d.add(String(x_label, height - 8, "Carga por dominios CKM/CRHM", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#0f172a")))
    for domain, score in scores.items():
        score = int(score)
        fill = colors.HexColor("#15803d") if score == 0 else colors.HexColor("#0ea5e9") if score == 1 else colors.HexColor("#b45309") if score == 2 else colors.HexColor("#b91c1c")
        d.add(String(x_label, y, pdf_clean_text(domain), fontName="Helvetica", fontSize=8.5, fillColor=colors.HexColor("#334155")))
        d.add(Rect(x_bar, y - 4, bar_w, 12, strokeColor=colors.HexColor("#cbd5e1"), fillColor=colors.HexColor("#f8fafc"), rx=2, ry=2))
        d.add(Rect(x_bar, y - 4, max(1, bar_w * score / 3.0), 12, strokeColor=fill, fillColor=fill, rx=2, ry=2))
        d.add(String(x_bar + bar_w + 10, y, f"{score}/3", fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.HexColor("#0f172a")))
        y -= row_h
    return d


def pdf_header_footer(canvas, doc) -> None:
    if not REPORTLAB_AVAILABLE:
        return
    canvas.saveState()
    page_w, page_h = A4
    canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, page_h - 1.15 * cm, page_w - doc.rightMargin, page_h - 1.15 * cm)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.setFillColor(colors.HexColor("#0f172a"))
    canvas.drawString(doc.leftMargin, page_h - 0.85 * cm, "Informe médico CKM/CRHM")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(page_w - doc.rightMargin, page_h - 0.85 * cm, datetime.now().strftime("%d/%m/%Y"))
    canvas.line(doc.leftMargin, 1.05 * cm, page_w - doc.rightMargin, 1.05 * cm)
    canvas.drawString(doc.leftMargin, 0.72 * cm, pdf_clean_text(AUTHOR))
    canvas.drawRightString(page_w - doc.rightMargin, 0.72 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def pdf_core_items(p: Dict[str, Any], r: Dict[str, Any]) -> List[List[Any]]:
    """Items clínicos principales para la tabla didáctica del PDF."""
    return [
        ["Adiposidad", f"IMC {fmt(r.get('bmi'))} kg/m2; cintura {p.get('waist_cm')} cm", f"{r.get('bmi_class')}; cintura {'aumentada' if r.get('abnormal_waist') else 'no aumentada'}"],
        ["Metabólico", f"Síndrome metabólico {r.get('metabolic_syndrome_count')}/5; HbA1c {p.get('hba1c')}%", "DM2" if r.get("t2d") else "Prediabetes" if r.get("prediabetes") else "Sin disglucemia detectada"],
        ["Presión arterial", f"PAS/PAD {p.get('sbp')}/{p.get('dbp')} mmHg", "HTA o PA elevada" if r.get("hypertension") else "Sin HTA detectada"],
        ["Renal", f"eGFR {p.get('egfr')} ml/min/1,73m2; UACR {p.get('uacr')} mg/g", f"{r.get('egfr_category')}-{r.get('albuminuria_category')} - KDIGO {r.get('kdigo_risk')}"],
        ["Hepático", f"MASLD {'sí' if r.get('masld_known') else 'no informado'}; FIB-4 {fmt(r.get('fib4'), 2)}", f"{r.get('fib4_category')}. {r.get('fib4_action')}"] ,
        ["Cardiovascular", f"PREVENT 10 años {fmt(r.get('prevent_10'))}%", "ECV clínica" if r.get("clinical_cvd") else "ECV subclínica" if r.get("subclinical_cvd") else "Sin ECV clínica/subclínica informada"],
    ]


def pdf_report_bytes(p: Dict[str, Any], r: Dict[str, Any], recos: List[Dict[str, str]]) -> bytes:
    """Genera un PDF profesional y didáctico del informe médico CKM/CRHM."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("Para generar PDF agregue reportlab a requirements.txt")

    buffer = io.BytesIO()
    page_w, _ = A4
    margin = 1.45 * cm
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title=f"Informe CKM CRHM {r.get('patient_code', 'Paciente')}",
        author=AUTHOR,
    )
    doc_width = page_w - 2 * margin

    base = getSampleStyleSheet()
    title_style = ParagraphStyle("PDFTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), alignment=TA_LEFT, spaceAfter=6)
    subtitle_style = ParagraphStyle("PDFSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=9.2, leading=12, textColor=colors.HexColor("#475569"), spaceAfter=8)
    section_style = ParagraphStyle("PDFSection", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.2, leading=15, textColor=colors.HexColor("#0f766e"), spaceBefore=10, spaceAfter=6)
    normal_style = ParagraphStyle("PDFNormal", parent=base["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.3, textColor=colors.HexColor("#0f172a"), spaceAfter=4)
    small_style = ParagraphStyle("PDFSmall", parent=base["Normal"], fontName="Helvetica", fontSize=7.7, leading=9.5, textColor=colors.HexColor("#475569"), spaceAfter=3)
    table_header_style = ParagraphStyle("PDFTableHeader", parent=normal_style, fontName="Helvetica-Bold", fontSize=8.0, leading=9.4, textColor=colors.white)
    table_cell_style = ParagraphStyle("PDFTableCell", parent=normal_style, fontSize=7.8, leading=9.2)

    story: List[Any] = []
    story.append(pdf_para("INFORME MÉDICO INTEGRADO", title_style))
    story.append(pdf_para("Síndrome cardiovascular-renal-metabólico (CKM) y cardio-reno-hepato-metabólico (CRHM)", subtitle_style))

    identification = [
        [pdf_rich("<b>Código del paciente</b>", table_cell_style), pdf_para(r.get("patient_code", "Paciente"), table_cell_style), pdf_rich("<b>Fecha</b>", table_cell_style), pdf_para(datetime.now().strftime("%d/%m/%Y %H:%M"), table_cell_style)],
        [pdf_rich("<b>Edad / sexo</b>", table_cell_style), pdf_para(f"{r.get('age')} años / {r.get('sex')}", table_cell_style), pdf_rich("<b>Autor</b>", table_cell_style), pdf_para(AUTHOR, table_cell_style)],
    ]
    t = Table(identification, colWidths=[3.0 * cm, 4.6 * cm, 2.1 * cm, doc_width - 9.7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 7))

    sem_color = colors.HexColor("#fee2e2") if "Muy" in r.get("traffic_light", "") or r.get("ckm_stage_n", 0) >= 4 else colors.HexColor("#fef3c7") if r.get("ckm_stage_n", 0) >= 2 else colors.HexColor("#dcfce7")
    conclusion_text = (
        f"<b>Conclusión operativa:</b> {pdf_clean_text(r.get('ckm_stage'))}. "
        f"{pdf_clean_text(r.get('crhm_label'))}. Semáforo clínico: <b>{pdf_clean_text(r.get('traffic_light'))}</b>. "
        "El objetivo es priorizar los dominios que impulsan la progresión y seleccionar terapias con beneficio transversal cardiovascular, renal, metabólico y hepático."
    )
    concl = Table([[pdf_rich(conclusion_text, normal_style)]], colWidths=[doc_width])
    concl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sem_color),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#94a3b8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(concl)

    story.append(pdf_para("1. Resultado integrado", section_style))
    result_rows = [
        [pdf_rich("<b>Etapa CKM</b>", table_cell_style), pdf_para(r.get("ckm_stage"), table_cell_style)],
        [pdf_rich("<b>Etapa CRHM</b>", table_cell_style), pdf_para(r.get("crhm_label"), table_cell_style)],
        [pdf_rich("<b>Semáforo</b>", table_cell_style), pdf_para(r.get("traffic_light"), table_cell_style)],
        [pdf_rich("<b>Riñón</b>", table_cell_style), pdf_para(f"{r.get('egfr_category')}-{r.get('albuminuria_category')} - KDIGO {r.get('kdigo_risk')}", table_cell_style)],
        [pdf_rich("<b>Hígado</b>", table_cell_style), pdf_para(f"FIB-4 {fmt(r.get('fib4'), 2)} ({r.get('fib4_category')})", table_cell_style)],
        [pdf_rich("<b>PREVENT ingresado</b>", table_cell_style), pdf_para(f"10 años {fmt(r.get('prevent_10'))}% - 30 años {fmt(r.get('prevent_30'))}%", table_cell_style)],
    ]
    rt = Table(result_rows, colWidths=[4.2 * cm, doc_width - 4.2 * cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfeff")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rt)

    story.append(pdf_para("2. Items clínicos de interés", section_style))
    items = [[pdf_rich("<b>Dominio</b>", table_header_style), pdf_rich("<b>Dato principal</b>", table_header_style), pdf_rich("<b>Interpretación</b>", table_header_style)]]
    for row in pdf_core_items(p, r):
        items.append([pdf_para(row[0], table_cell_style), pdf_para(row[1], table_cell_style), pdf_para(row[2], table_cell_style)])
    it = Table(items, colWidths=[3.0 * cm, 5.2 * cm, doc_width - 8.2 * cm], repeatRows=1)
    it.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(it)

    story.append(pdf_para("3. Mapa didáctico de carga por dominios", section_style))
    story.append(pdf_domain_drawing(r.get("domain_scores", {})))
    story.append(pdf_para("Escala: 0 = sin carga evidente; 1 = leve; 2 = moderada; 3 = alta. El dominio con mayor carga debe orientar la prioridad del seguimiento y de la intervención.", small_style))

    story.append(PageBreak())
    story.append(pdf_para("4. Propuesta terapéutica integrada", section_style))
    story.append(pdf_para("Las recomendaciones son operativas y requieren confirmación de indicación, dosis, contraindicaciones, interacciones, función renal, potasio, cobertura y preferencias del paciente.", small_style))

    rec_table = [[
        pdf_rich("<b>Prioridad</b>", table_header_style),
        pdf_rich("<b>Dominio</b>", table_header_style),
        pdf_rich("<b>Recomendación</b>", table_header_style),
        pdf_rich("<b>Seguimiento</b>", table_header_style),
    ]]
    for rec in recos:
        rec_table.append([
            pdf_para(rec.get("Prioridad", ""), table_cell_style),
            pdf_para(rec.get("Dominio", ""), table_cell_style),
            pdf_para(rec.get("Recomendación", ""), table_cell_style),
            pdf_para(rec.get("Seguimiento", ""), table_cell_style),
        ])
    rec_t = Table(rec_table, colWidths=[2.0 * cm, 3.0 * cm, 7.1 * cm, doc_width - 12.1 * cm], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for idx, rec in enumerate(recos, start=1):
        style_cmds.append(("BACKGROUND", (0, idx), (0, idx), pdf_priority_color(rec.get("Prioridad", ""))))
    rec_t.setStyle(TableStyle(style_cmds))
    story.append(rec_t)

    story.append(pdf_para("5. Plan de control sugerido", section_style))
    control_rows = [
        ["4-12 semanas", "PA, peso, cintura, adherencia, tolerancia y eventos adversos; revisar automonitoreo."],
        ["3-6 meses", "HbA1c/glucemia, lípidos, eGFR, UACR, potasio si usa SRAA/MRA, ajuste terapéutico por objetivos."],
        ["6-12 meses", "Recalcular etapa CKM/CRHM, revisar PREVENT si corresponde, reevaluar MASLD/FIB-4 y necesidad de elastografía."],
        ["Derivación", "Nefrología si ERC progresiva/alto riesgo; hepatología si FIB-4 alto, elastografía alterada o sospecha de cirrosis; cardiología si ECV clínica, pre-IC/IC o alto riesgo."],
    ]
    control = [[pdf_rich("<b>Momento</b>", table_header_style), pdf_rich("<b>Acción</b>", table_header_style)]] + [[pdf_para(a, table_cell_style), pdf_para(b, table_cell_style)] for a, b in control_rows]
    ct = Table(control, colWidths=[3.0 * cm, doc_width - 3.0 * cm], repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ct)

    story.append(Spacer(1, 8))
    disclaimer = (
        "Este informe es una herramienta de apoyo clínico y educación médica. No reemplaza el juicio clínico, "
        "la confirmación diagnóstica ni la prescripción individualizada. Antes de indicar tratamiento confirmar dosis, "
        "contraindicaciones, interacciones, función renal, potasio, estado hepático, embarazo/lactancia cuando aplique, "
        "acceso/cobertura y preferencias del paciente."
    )
    story.append(pdf_para(disclaimer, small_style))

    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)
    buffer.seek(0)
    return buffer.getvalue()


def to_excel_bytes(summary: Dict[str, Any], recos: List[Dict[str, str]]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Resumen", index=False)
        pd.DataFrame(recos).to_excel(writer, sheet_name="Recomendaciones", index=False)
    bio.seek(0)
    return bio.getvalue()



# -----------------------------------------------------------------------------
# Importación de PDFs de laboratorio
# -----------------------------------------------------------------------------
def _norm_lab_text(text: str) -> str:
    """Normaliza texto de laboratorio para búsquedas por regex."""
    text = text or ""
    text = text.replace("\r", "\n")
    # Unificar separadores frecuentes sin borrar saltos de línea.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _num_from_match(value: Any) -> Optional[float]:
    """Convierte números tipo 1.234,5 o 1234.5 a float."""
    if value is None:
        return None
    s = str(value).strip()
    s = s.replace(" ", "")
    # Evitar que puntos de miles se interpreten como decimal cuando también hay coma.
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _line_value_search(text: str, labels: List[str], min_v: float, max_v: float) -> Tuple[Optional[float], str]:
    """
    Busca un valor numérico en líneas que contienen cualquiera de las etiquetas.
    Devuelve valor y línea fuente para auditoría visual.
    """
    lines = [ln.strip() for ln in _norm_lab_text(text).splitlines() if ln.strip()]
    label_re = re.compile("|".join(labels), flags=re.I)
    # Buscar primero líneas completas. El primer número plausible posterior a la etiqueta suele ser el resultado.
    for ln in lines:
        if not label_re.search(ln):
            continue
        # Evitar que LDL detecte HDL si la línea tiene ambos por encabezado: se controla desde labels específicas.
        numbers = re.findall(r"(?<![A-Za-z])[-+]?\d{1,5}(?:[\.,]\d+)?", ln)
        for n in numbers:
            v = _num_from_match(n)
            if v is not None and min_v <= v <= max_v:
                return v, ln[:180]
    # Si viene en bloque sin saltos correctos, buscar patrón etiqueta...número.
    pat = re.compile(r"(?:" + "|".join(labels) + r")[^\n\d]{0,60}([-+]?\d{1,5}(?:[\.,]\d+)?)", flags=re.I)
    for m in pat.finditer(text):
        v = _num_from_match(m.group(1))
        if v is not None and min_v <= v <= max_v:
            return v, text[max(0, m.start()-30): min(len(text), m.end()+60)].replace("\n", " ")[:180]
    return None, ""


def egfr_ckd_epi_2021(creatinine_mg_dl: Optional[float], age: Optional[float], sex: str) -> Optional[float]:
    """eGFR CKD-EPI 2021 sin raza, útil si el laboratorio informa creatinina pero no eGFR."""
    if not is_number(creatinine_mg_dl) or not is_number(age) or creatinine_mg_dl <= 0 or age <= 0:
        return None
    female = str(sex).lower().startswith("muj")
    k = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    scr_k = float(creatinine_mg_dl) / k
    egfr = 142 * (min(scr_k, 1) ** alpha) * (max(scr_k, 1) ** -1.200) * (0.9938 ** float(age))
    if female:
        egfr *= 1.012
    return round(egfr, 1)


def extract_text_from_lab_pdf(uploaded_file: Any) -> Tuple[str, List[str]]:
    """Extrae texto de un PDF de laboratorio digital. No realiza OCR."""
    warnings: List[str] = []
    if not PYPDF_AVAILABLE:
        return "", ["No está instalado pypdf. Agregue pypdf>=4.0 a requirements.txt."]
    try:
        raw = uploaded_file.getvalue()
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                warnings.append(f"No se pudo leer texto de la página {i+1}.")
        text = _norm_lab_text("\n".join(pages))
        if len(text.strip()) < 80:
            warnings.append("El PDF parece escaneado o contiene texto no extraíble. Para esos casos se requiere OCR previo o cargar los datos manualmente.")
        return text, warnings
    except Exception as e:
        return "", [f"No se pudo leer el PDF: {e}"]


def parse_lab_pdf_text(text: str, age: Optional[float] = None, sex: str = "Hombre") -> Dict[str, Any]:
    """Detecta variables útiles para CKM/CRHM desde el texto del laboratorio."""
    text = _norm_lab_text(text)
    patterns = {
        "fpg": ([r"\bglucemia\b", r"\bglucosa\b", r"\bglicemia\b", r"\bglucose\b"], 40, 600, "Glucemia", "mg/dL"),
        "hba1c": ([r"hba1c", r"hb\s*a1c", r"hemoglobina\s+glicosilada", r"hemoglobina\s+glucosilada", r"a1c"], 3, 18, "HbA1c", "%"),
        "tg": ([r"trigliceridos", r"triglic[eé]ridos", r"triglycerides", r"\btg\b"], 20, 3000, "Triglicéridos", "mg/dL"),
        "hdl": ([r"hdl(?:[- ]?c)?", r"colesterol\s+hdl"], 5, 200, "HDL-C", "mg/dL"),
        "ldl": ([r"\bldl(?:[- ]?c)?\b", r"colesterol\s+ldl"], 1, 500, "LDL-C", "mg/dL"),
        "egfr": ([r"e\s*gf\s*r", r"egfr", r"filtrado\s+glomerular", r"tfg", r"clearance\s+estimado"], 1, 180, "eGFR/TFG", "mL/min/1,73m2"),
        "uacr": ([r"uacr", r"cacu", r"albumina\s*/\s*creatinina", r"albuminuria", r"microalbuminuria"], 0, 10000, "UACR/CACU", "mg/g"),
        "ast": ([r"\bast\b", r"\btgo\b", r"aspartato\s+aminotransferasa", r"got\b"], 1, 2000, "AST/TGO", "U/L"),
        "alt": ([r"\balt\b", r"\btgp\b", r"alanina\s+aminotransferasa", r"gpt\b"], 1, 2000, "ALT/TGP", "U/L"),
        "platelets": ([r"plaquetas", r"recuento\s+plaquetario", r"platelets", r"plt\b"], 5, 1500, "Plaquetas", "10^9/L"),
        "creatinine": ([r"creatinina", r"creatinine", r"\bscr\b"], 0.2, 20, "Creatinina", "mg/dL"),
        "total_cholesterol": ([r"colesterol\s+total", r"cholesterol\s+total"], 50, 600, "Colesterol total", "mg/dL"),
        "non_hdl": ([r"no\s*hdl", r"non\s*hdl"], 20, 500, "No-HDL", "mg/dL"),
        "uric_acid": ([r"acido\s+urico", r"ácido\s+úrico", r"uric\s+acid"], 1, 20, "Ácido úrico", "mg/dL"),
        "ggt": ([r"\bggt\b", r"gamma\s*gt", r"gammaglutamil"], 1, 2000, "GGT", "U/L"),
    }
    extracted: Dict[str, Any] = {}
    audit_rows: List[Dict[str, Any]] = []
    for key, (labels, min_v, max_v, label, unit) in patterns.items():
        val, source = _line_value_search(text, labels, min_v, max_v)
        if val is not None:
            extracted[key] = val
            audit_rows.append({"Variable": label, "Valor detectado": val, "Unidad sugerida": unit, "Línea fuente": source})

    # Plaquetas a veces se informan como 210000/mm3; convertir a 210 x10^9/L.
    if "platelets" not in extracted:
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            if re.search(r"plaquetas|recuento\s+plaquetario|platelets|\bplt\b", ln, flags=re.I):
                nums = re.findall(r"\d{5,6}(?:[\.,]\d+)?|\d{1,3}(?:[\.,]\d{3})", ln)
                for n in nums:
                    val = _num_from_match(n)
                    if val is not None and val > 1500:
                        val = round(val / 1000.0, 1)
                    if val is not None and 5 <= val <= 1500:
                        extracted["platelets"] = val
                        audit_rows.append({"Variable": "Plaquetas", "Valor detectado": val, "Unidad sugerida": "10^9/L", "Línea fuente": ln[:180]})
                        break
            if "platelets" in extracted:
                break

    # Si no se detecta UACR, intentar patrones frecuentes con mg/g explícito alrededor.
    if "uacr" not in extracted:
        m = re.search(r"(?:microalbuminuria|albuminuria)[^\n]{0,90}?([-+]?\d{1,5}(?:[\.,]\d+)?)\s*(?:mg\s*/\s*g|mg/g)", text, flags=re.I)
        if m:
            val = _num_from_match(m.group(1))
            if val is not None:
                extracted["uacr"] = val
                audit_rows.append({"Variable": "UACR/CACU", "Valor detectado": val, "Unidad sugerida": "mg/g", "Línea fuente": m.group(0)[:180]})

    # Calcular eGFR si hay creatinina y no vino eGFR informada.
    if "egfr" not in extracted and "creatinine" in extracted and age is not None:
        calc = egfr_ckd_epi_2021(extracted.get("creatinine"), age, sex)
        if calc is not None:
            extracted["egfr"] = calc
            audit_rows.append({"Variable": "eGFR estimada CKD-EPI 2021", "Valor detectado": calc, "Unidad sugerida": "mL/min/1,73m2", "Línea fuente": "Calculada desde creatinina, edad y sexo ingresados"})

    # Sugerir MASLD probable si ALT/GGT elevadas con patrón metabólico; no lo fuerza automáticamente.
    extracted["audit_rows"] = audit_rows
    if text:
        extracted["patient_code"] = "LAB-" + hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8].upper()
    return extracted


def apply_lab_values_to_session(extracted: Dict[str, Any]) -> None:
    """Carga valores extraídos en session_state para prellenar los widgets."""
    key_map = {
        "patient_code": "ckm_patient_code",
        "fpg": "ckm_fpg",
        "hba1c": "ckm_hba1c",
        "tg": "ckm_tg",
        "hdl": "ckm_hdl",
        "ldl": "ckm_ldl",
        "egfr": "ckm_egfr",
        "uacr": "ckm_uacr",
        "ast": "ckm_ast",
        "alt": "ckm_alt",
        "platelets": "ckm_platelets",
    }
    for src, dst in key_map.items():
        if src in extracted and extracted[src] not in [None, ""]:
            st.session_state[dst] = extracted[src]
    st.session_state["ckm_lab_last_applied"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def render_lab_pdf_importer() -> None:
    """Widget de importación PDF en la barra lateral."""
    with st.expander("0) Importar PDF de laboratorio", expanded=True):
        st.caption("Importa PDF digital de laboratorio y prellena glucemia, HbA1c, lípidos, eGFR/UACR, AST/ALT y plaquetas. Si el PDF es escaneado, requiere OCR previo.")
        uploaded_pdf = st.file_uploader("PDF de resultados de laboratorio", type=["pdf"], key="lab_pdf_uploader")
        if uploaded_pdf is None:
            if st.session_state.get("ckm_lab_last_applied"):
                st.success(f"Último PDF aplicado: {st.session_state['ckm_lab_last_applied']}")
            return
        text, warnings = extract_text_from_lab_pdf(uploaded_pdf)
        for w in warnings:
            st.warning(w)
        extracted = parse_lab_pdf_text(
            text,
            age=st.session_state.get("ckm_age", 58),
            sex=st.session_state.get("ckm_sex", "Hombre"),
        )
        st.session_state["ckm_last_lab_text"] = text
        st.session_state["ckm_last_lab_extracted"] = extracted
        rows = extracted.get("audit_rows", [])
        if rows:
            st.success(f"Se detectaron {len(rows)} variables potencialmente útiles.")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=180)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Aplicar al formulario", type="primary", use_container_width=True):
                    apply_lab_values_to_session(extracted)
                    st.rerun()
            with col_b:
                st.download_button(
                    "Descargar texto",
                    data=(text or "").encode("utf-8"),
                    file_name=f"texto_laboratorio_{extracted.get('patient_code', 'LAB')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.info("No se detectaron variables con seguridad. Puede revisar el texto extraído o cargar los datos manualmente.")
            with st.expander("Ver texto extraído"):
                st.text_area("Texto del PDF", value=text[:12000], height=250)

# -----------------------------------------------------------------------------
# Interfaz: inputs
# -----------------------------------------------------------------------------
def patient_inputs() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🧭 Ingreso clínico")
        st.caption("Datos mínimos: edad, sexo, peso/talla/cintura, PA, glucosa/HbA1c, lípidos, eGFR/UACR, AST/ALT/plaquetas.")
        render_lab_pdf_importer()

        with st.expander("1) Identificación no nominal", expanded=True):
            patient_code = st.text_input("Código del paciente", value=st.session_state.get("ckm_patient_code", "CKM-001"), key="ckm_patient_code", help="Evitar nombre y DNI en exportaciones.")
            age = st.number_input("Edad (años)", min_value=18, max_value=110, value=int(st.session_state.get("ckm_age", 58)), step=1, key="ckm_age")
            sex = st.selectbox("Sexo", ["Hombre", "Mujer"], index=0 if st.session_state.get("ckm_sex", "Hombre") == "Hombre" else 1, key="ckm_sex")
            asian = st.checkbox("Ascendencia asiática (ajusta cintura/IMC)", value=bool(st.session_state.get("ckm_asian_ancestry", False)), key="ckm_asian_ancestry")

        with st.expander("2) Antropometría", expanded=True):
            weight_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=float(st.session_state.get("ckm_weight_kg", 92.0)), step=0.5, key="ckm_weight_kg")
            height_cm = st.number_input("Talla (cm)", min_value=120.0, max_value=220.0, value=float(st.session_state.get("ckm_height_cm", 172.0)), step=0.5, key="ckm_height_cm")
            waist_cm = st.number_input("Cintura (cm)", min_value=40.0, max_value=220.0, value=float(st.session_state.get("ckm_waist_cm", 108.0)), step=0.5, key="ckm_waist_cm")

        with st.expander("3) Presión arterial / riesgo", expanded=True):
            sbp = st.number_input("PAS consultorio o promedio domiciliario", min_value=70, max_value=260, value=int(st.session_state.get("ckm_sbp", 138)), step=1, key="ckm_sbp")
            dbp = st.number_input("PAD consultorio o promedio domiciliario", min_value=40, max_value=160, value=int(st.session_state.get("ckm_dbp", 86)), step=1, key="ckm_dbp")
            on_bp_meds = st.checkbox("Recibe antihipertensivos", value=bool(st.session_state.get("ckm_on_bp_meds", True)), key="ckm_on_bp_meds")
            prevent_10 = st.number_input("PREVENT 10 años ECV total (%) - opcional", min_value=0.0, max_value=100.0, value=float(st.session_state.get("ckm_prevent_10", 12.0)), step=0.5, key="ckm_prevent_10")
            prevent_30 = st.number_input("PREVENT 30 años ECV total (%) - opcional", min_value=0.0, max_value=100.0, value=float(st.session_state.get("ckm_prevent_30", 35.0)), step=0.5, key="ckm_prevent_30")

        with st.expander("4) Laboratorio metabólico", expanded=True):
            fpg = st.number_input("Glucemia en ayunas (mg/dL)", min_value=40.0, max_value=600.0, value=float(st.session_state.get("ckm_fpg", 112.0)), step=1.0, key="ckm_fpg")
            ogtt_2h = st.number_input("PTOG 2 h (mg/dL, opcional)", min_value=0.0, max_value=600.0, value=float(st.session_state.get("ckm_ogtt_2h", 0.0)), step=1.0, key="ckm_ogtt_2h", help="0 = no disponible")
            hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=16.0, value=float(st.session_state.get("ckm_hba1c", 6.1)), step=0.1, key="ckm_hba1c")
            known_t2d = st.checkbox("Diabetes tipo 2 conocida", value=bool(st.session_state.get("ckm_known_t2d", False)), key="ckm_known_t2d")
            tg = st.number_input("Triglicéridos (mg/dL)", min_value=30.0, max_value=2000.0, value=float(st.session_state.get("ckm_tg", 190.0)), step=1.0, key="ckm_tg")
            hdl = st.number_input("HDL-C (mg/dL)", min_value=10.0, max_value=150.0, value=float(st.session_state.get("ckm_hdl", 38.0)), step=1.0, key="ckm_hdl")
            ldl = st.number_input("LDL-C (mg/dL)", min_value=0.0, max_value=400.0, value=float(st.session_state.get("ckm_ldl", 126.0)), step=1.0, key="ckm_ldl")

        with st.expander("5) Riñón", expanded=True):
            egfr = st.number_input("eGFR (mL/min/1,73m²)", min_value=0.0, max_value=150.0, value=float(st.session_state.get("ckm_egfr", 68.0)), step=1.0, key="ckm_egfr")
            uacr = st.number_input("UACR/CACU (mg/g)", min_value=0.0, max_value=5000.0, value=float(st.session_state.get("ckm_uacr", 45.0)), step=1.0, key="ckm_uacr")
            dialysis = st.checkbox("Diálisis", value=bool(st.session_state.get("ckm_dialysis", False)), key="ckm_dialysis")
            kidney_transplant = st.checkbox("Trasplante renal", value=bool(st.session_state.get("ckm_kidney_transplant", False)), key="ckm_kidney_transplant")

        with st.expander("6) Hígado / MASLD", expanded=True):
            masld_known = st.checkbox("Esteatosis hepática/MASLD conocida por imagen o informe", value=bool(st.session_state.get("ckm_masld_known", True)), key="ckm_masld_known")
            liver_stage_options = ["Desconocido", "F0-F1", "F2-F4", "Cirrhosis"]
            liver_stage_default = st.session_state.get("ckm_liver_stage", "Desconocido")
            liver_stage = st.selectbox("Fibrosis hepática conocida", liver_stage_options, index=liver_stage_options.index(liver_stage_default) if liver_stage_default in liver_stage_options else 0, key="ckm_liver_stage")
            ast = st.number_input("AST/TGO (U/L)", min_value=0.0, max_value=1000.0, value=float(st.session_state.get("ckm_ast", 34.0)), step=1.0, key="ckm_ast")
            alt = st.number_input("ALT/TGP (U/L)", min_value=0.0, max_value=1000.0, value=float(st.session_state.get("ckm_alt", 42.0)), step=1.0, key="ckm_alt")
            platelets = st.number_input("Plaquetas (10⁹/L)", min_value=1.0, max_value=1000.0, value=float(st.session_state.get("ckm_platelets", 240.0)), step=1.0, key="ckm_platelets")

        with st.expander("7) Cardiovascular", expanded=True):
            ascvd = st.checkbox("ASCVD clínica: IAM/angina/revascularización/coronaria", value=bool(st.session_state.get("ckm_ascvd", False)), key="ckm_ascvd")
            stroke_tia = st.checkbox("ACV/AIT", value=bool(st.session_state.get("ckm_stroke_tia", False)), key="ckm_stroke_tia")
            pad = st.checkbox("Enfermedad arterial periférica", value=bool(st.session_state.get("ckm_pad", False)), key="ckm_pad")
            hf = st.checkbox("Insuficiencia cardíaca clínica", value=bool(st.session_state.get("ckm_hf", False)), key="ckm_hf")
            af = st.checkbox("Fibrilación auricular", value=bool(st.session_state.get("ckm_af", False)), key="ckm_af")
            cac_gt100 = st.checkbox("CAC >100 o aterosclerosis subclínica relevante", value=bool(st.session_state.get("ckm_cac_gt100", False)), key="ckm_cac_gt100")
            subclinical_atherosclerosis = st.checkbox("Placa carotídea/femoral/aórtica subclínica", value=bool(st.session_state.get("ckm_subclinical_atherosclerosis", False)), key="ckm_subclinical_atherosclerosis")
            pre_hf = st.checkbox("Pre-IC: NT-proBNP/troponina/eco anormal sin IC clínica", value=bool(st.session_state.get("ckm_pre_hf", False)), key="ckm_pre_hf")
            osa_suspected = st.checkbox("Sospecha de apnea del sueño", value=bool(st.session_state.get("ckm_osa_suspected", False)), key="ckm_osa_suspected")

    ogtt_value = None if ogtt_2h == 0 else ogtt_2h
    return {
        "patient_code": patient_code,
        "age": age,
        "sex": sex,
        "asian_ancestry": asian,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "waist_cm": waist_cm,
        "sbp": sbp,
        "dbp": dbp,
        "on_bp_meds": on_bp_meds,
        "prevent_10": prevent_10,
        "prevent_30": prevent_30,
        "fpg": fpg,
        "ogtt_2h": ogtt_value,
        "hba1c": hba1c,
        "known_t2d": known_t2d,
        "tg": tg,
        "hdl": hdl,
        "ldl": ldl,
        "egfr": egfr,
        "uacr": uacr,
        "dialysis": dialysis,
        "kidney_transplant": kidney_transplant,
        "masld_known": masld_known,
        "liver_stage": liver_stage,
        "ast": ast,
        "alt": alt,
        "platelets": platelets,
        "ascvd": ascvd,
        "stroke_tia": stroke_tia,
        "pad": pad,
        "hf": hf,
        "af": af,
        "cac_gt100": cac_gt100,
        "subclinical_atherosclerosis": subclinical_atherosclerosis,
        "pre_hf": pre_hf,
        "osa_suspected": osa_suspected,
    }


# -----------------------------------------------------------------------------
# Batch CSV
# -----------------------------------------------------------------------------
TEMPLATE_COLUMNS = [
    "patient_code", "age", "sex", "weight_kg", "height_cm", "waist_cm", "sbp", "dbp", "on_bp_meds",
    "fpg", "hba1c", "ogtt_2h", "known_t2d", "tg", "hdl", "ldl", "egfr", "uacr", "ast", "alt", "platelets",
    "masld_known", "liver_stage", "ascvd", "stroke_tia", "pad", "hf", "af", "cac_gt100", "subclinical_atherosclerosis", "pre_hf", "osa_suspected", "prevent_10", "prevent_30",
]


def template_csv_bytes() -> bytes:
    # Pandas/pyarrow en Streamlit Cloud puede inferir columnas string y fallar
    # al asignar valores numéricos luego con df.loc. Por eso la fila se crea
    # completa desde el inicio, manteniendo tipos mixtos compatibles.
    example = {col: "" for col in TEMPLATE_COLUMNS}
    example.update({
        "patient_code": "CKM-001",
        "age": 58,
        "sex": "Hombre",
        "weight_kg": 92,
        "height_cm": 172,
        "waist_cm": 108,
        "sbp": 138,
        "dbp": 86,
        "on_bp_meds": True,
        "fpg": 112,
        "hba1c": 6.1,
        "ogtt_2h": "",
        "known_t2d": False,
        "tg": 190,
        "hdl": 38,
        "ldl": 126,
        "egfr": 68,
        "uacr": 45,
        "ast": 34,
        "alt": 42,
        "platelets": 240,
        "masld_known": True,
        "liver_stage": "Desconocido",
        "ascvd": False,
        "stroke_tia": False,
        "pad": False,
        "hf": False,
        "af": False,
        "cac_gt100": False,
        "subclinical_atherosclerosis": False,
        "pre_hf": False,
        "osa_suspected": False,
        "prevent_10": 12,
        "prevent_30": 35,
    })
    df = pd.DataFrame([example], columns=TEMPLATE_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ["true", "1", "si", "sí", "yes", "y", "verdadero"]


def row_to_patient(row: pd.Series) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    for col in TEMPLATE_COLUMNS:
        p[col] = row[col] if col in row.index else None
    for bcol in ["on_bp_meds", "known_t2d", "masld_known", "ascvd", "stroke_tia", "pad", "hf", "af", "cac_gt100", "subclinical_atherosclerosis", "pre_hf", "osa_suspected"]:
        p[bcol] = coerce_bool(p.get(bcol))
    if p.get("sex") not in ["Hombre", "Mujer"]:
        p["sex"] = "Hombre"
    p["asian_ancestry"] = False
    p["dialysis"] = False
    p["kidney_transplant"] = False
    p["subclinical_atherosclerosis"] = bool(p.get("subclinical_atherosclerosis", False))
    p["osa_suspected"] = bool(p.get("osa_suspected", False))
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    st.markdown(
        f"""
        <div class='app-hero'>
            <div class='main-title'>🫀 {APP_TITLE}</div>
            <div class='subtitle'><b>{APP_SUBTITLE}</b><br>
                Herramienta clínica operativa para estadificación CKM, extensión CRHM, FIB-4, KDIGO,
                síndrome metabólico y propuesta terapéutica integrada.<br>
                <b>{AUTHOR}</b> · {VERSION}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p = patient_inputs()
    r = evaluate_patient(p)
    recos = generate_recommendations(p, r)

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 PDF laboratorio",
        "📌 Resultado integrado",
        "🧬 Dominios y fisiopatología",
        "💊 Manejo terapéutico",
        "📥 Exportar / carga masiva",
        "📚 Base de conocimiento",
    ])

    with tab0:
        st.markdown("### Importación de PDF de laboratorio")
        st.info("Use el cargador de la barra lateral para importar un PDF digital. La app detecta y aplica variables clave para CKM/CRHM: glucemia, HbA1c, TG, HDL, LDL, eGFR, UACR, AST/TGO, ALT/TGP y plaquetas. Los valores quedan siempre editables en el formulario clínico.")
        extracted = st.session_state.get("ckm_last_lab_extracted", {})
        rows = extracted.get("audit_rows", []) if isinstance(extracted, dict) else []
        if rows:
            st.markdown("#### Variables detectadas en el último PDF")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if "ckm_last_lab_text" in st.session_state:
                with st.expander("Ver texto extraído del PDF"):
                    st.text_area("Texto extraído", value=st.session_state.get("ckm_last_lab_text", "")[:20000], height=320)
        else:
            st.warning("Todavía no hay un PDF de laboratorio importado o no se detectaron variables con seguridad.")

    with tab1:
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-label'>Etapa CKM</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi'>{r['ckm_stage_n']}/4</div>", unsafe_allow_html=True)
            st.write(r["ckm_stage"])
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-label'>Etapa CRHM</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi'>{r['crhm_stage']}</div>", unsafe_allow_html=True)
            st.write(r["crhm_label"])
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-label'>Semáforo</div>", unsafe_allow_html=True)
            st.markdown(render_badge(r["traffic_light"], r["traffic_light_cls"]), unsafe_allow_html=True)
            st.write("Intensidad clínica sugerida")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-label'>FIB-4</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi'>{fmt(r['fib4'], 2)}</div>", unsafe_allow_html=True)
            st.write(r["fib4_category"])
            st.markdown("</div>", unsafe_allow_html=True)

        cc1, cc2 = st.columns([1, 1.4])
        with cc1:
            st.markdown("#### Etapa CKM")
            render_stage_chart(r["ckm_stage_n"])
        with cc2:
            st.markdown("#### Mapa de carga por dominios CKM/CRHM")
            render_domain_chart(r["domain_scores"])

        st.markdown("### Interpretación clínica resumida")
        summary_points = [
            f"**Adiposidad:** IMC {fmt(r['bmi'])} kg/m² ({r['bmi_class']}); cintura {'aumentada' if r['abnormal_waist'] else 'no aumentada'} para umbral {r['waist_threshold']:.0f} cm.",
            f"**Metabolismo:** síndrome metabólico {r['metabolic_syndrome_count']}/5 criterios; {'DM2' if r['t2d'] else 'prediabetes' if r['prediabetes'] else 'sin disglucemia detectada'}.",
            f"**Riñón:** {r['egfr_category']}-{r['albuminuria_category']}, riesgo KDIGO {r['kdigo_risk']}.",
            f"**Hígado:** MASLD {'sí' if r['masld_known'] else 'no informado'}; FIB-4 {fmt(r['fib4'], 2)} ({r['fib4_category']}). {r['fib4_action']}",
            f"**Cardiovascular:** {'ECV clínica' if r['clinical_cvd'] else 'ECV subclínica' if r['subclinical_cvd'] else 'sin ECV clínica/subclínica informada'}; PREVENT 10 años {fmt(r['prevent_10'])}%.",
        ]
        for point in summary_points:
            st.markdown(f"- {point}")

        with st.expander("Criterios de síndrome metabólico detectados"):
            df_ms = pd.DataFrame([{"Criterio": k, "Presente": "Sí" if v else "No"} for k, v in r["metabolic_syndrome_criteria"].items()])
            st.dataframe(df_ms, hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("### Dominios CKM/CRHM")
        dcols = st.columns(5)
        for idx, (domain, score) in enumerate(r["domain_scores"].items()):
            with dcols[idx]:
                cls = "badge-ok" if score == 0 else "badge-info" if score == 1 else "badge-warn" if score == 2 else "badge-danger"
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{domain}</h3>", unsafe_allow_html=True)
                st.markdown(render_badge(f"Carga {score}/3", cls), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Lectura fisiopatológica")
        st.markdown(
            """
La app interpreta el cuadro como una red multiorgánica: adiposidad/insulino-resistencia, presión arterial,
aterogénesis, albuminuria/filtrado glomerular, esteatosis/fibrosis hepática y daño cardiovascular clínico o subclínico.
El objetivo práctico no es solo “nombrar” el síndrome, sino identificar cuál es el dominio que más empuja la progresión y
qué terapias tienen beneficio transversal.
            """
        )

        st.markdown("### Variables derivadas")
        derived = pd.DataFrame([
            ["IMC", fmt(r["bmi"]), r["bmi_class"]],
            ["Cintura", f"{p['waist_cm']} cm", "Aumentada" if r["abnormal_waist"] else "No aumentada"],
            ["Síndrome metabólico", f"{r['metabolic_syndrome_count']}/5", "Presente" if r["metabolic_syndrome"] else "No"],
            ["KDIGO", f"{r['egfr_category']}-{r['albuminuria_category']}", r["kdigo_risk"]],
            ["FIB-4", fmt(r["fib4"], 2), r["fib4_category"]],
            ["PREVENT 10 años", f"{fmt(r['prevent_10'])}%", "≥20%: equivalente etapa 3" if is_number(r["prevent_10"]) and r["prevent_10"] >= 20 else "<20%"],
        ], columns=["Variable", "Valor", "Interpretación"])
        st.dataframe(derived, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("### Propuesta terapéutica integrada")
        st.caption("Ordenada por prioridad clínica. Confirmar indicaciones, contraindicaciones, dosis, interacciones, cobertura y preferencias del paciente.")
        render_recommendations_chart(recos)
        st.markdown("---")
        for rec in recos:
            css = "reco-high" if rec["Prioridad"] == "Alta" else "reco-mid" if rec["Prioridad"] == "Media" else "reco-low"
            st.markdown(f"<div class='card {css}'>", unsafe_allow_html=True)
            st.markdown(f"### {rec['Dominio']} · prioridad {rec['Prioridad']}")
            st.markdown(f"**Recomendación:** {rec['Recomendación']}")
            st.markdown(f"**Fundamento:** {rec['Fundamento']}")
            st.markdown(f"**Seguimiento:** {rec['Seguimiento']}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Checklist de seguridad antes de prescribir")
        st.checkbox("Revisé eGFR, UACR, potasio y presión arterial antes de SRAA/SGLT2i/nsMRA.")
        st.checkbox("Revisé contraindicaciones, interacciones y tolerancia para GLP-1/GIP-GLP-1.")
        st.checkbox("Definí objetivos de PA, peso, LDL/no-HDL/ApoB, HbA1c y albuminuria.")
        st.checkbox("Planifiqué seguimiento y educación del paciente.")

    with tab4:
        st.markdown("### Exportar evaluación individual")
        summary = flatten_summary(p, r)
        md_report = report_markdown(p, r, recos)
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = pdf_report_bytes(p, r, recos)
                st.download_button(
                    "⬇️ Descargar informe PDF médico",
                    data=pdf_bytes,
                    file_name=f"informe_medico_CKM_CRHM_{safe_file_part(r['patient_code'])}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"No se pudo generar el PDF: {e}")
        else:
            st.warning("Para generar PDF agregá reportlab>=4.0 en requirements.txt y reiniciá la app.")
        st.download_button("⬇️ Descargar informe Markdown", data=md_report.encode("utf-8"), file_name=f"informe_CKM_CRHM_{safe_file_part(r['patient_code'])}.md", mime="text/markdown")
        st.download_button("⬇️ Descargar Excel", data=to_excel_bytes(summary, recos), file_name=f"informe_CKM_CRHM_{safe_file_part(r['patient_code'])}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("⬇️ Descargar JSON", data=json.dumps({"resumen": summary, "recomendaciones": recos}, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"informe_CKM_CRHM_{safe_file_part(r['patient_code'])}.json", mime="application/json")

        st.markdown("---")
        st.markdown("### Evaluación masiva CSV")
        st.download_button("Descargar plantilla CSV", data=template_csv_bytes(), file_name="plantilla_ckm_crhm.csv", mime="text/csv")
        up = st.file_uploader("Cargar CSV con columnas de la plantilla", type=["csv"])
        if up is not None:
            try:
                df_in = pd.read_csv(up)
                rows = []
                for _, row in df_in.iterrows():
                    rp = row_to_patient(row)
                    rr = evaluate_patient(rp)
                    rows.append(flatten_summary(rp, rr))
                df_out = pd.DataFrame(rows)
                st.success(f"Procesados {len(df_out)} pacientes.")
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    df_out.to_excel(writer, index=False, sheet_name="Resultados")
                st.download_button("⬇️ Descargar resultados masivos Excel", data=out.getvalue(), file_name="resultados_masivos_ckm_crhm.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"No se pudo procesar el CSV: {e}")

    with tab5:
        st.markdown("### Base de conocimiento implementada")
        st.markdown(
            """
**Importación PDF laboratorio**: detecta texto de PDFs digitales y prellena automáticamente variables de laboratorio para acelerar la evaluación CKM/CRHM. En PDFs escaneados se requiere OCR previo.

**CKM**: la app clasifica etapa 0 a 4, integra adiposidad, prediabetes/DM2, HTA, hipertrigliceridemia,
síndrome metabólico, ERC por eGFR/UACR, ECV subclínica, ECV clínica y el riesgo PREVENT ingresado.

**CRHM**: agrega el dominio hepático, con MASLD y FIB-4 como puerta de entrada para detectar fibrosis y decidir elastografía/derivación.

**Terapias transversales**: el motor sugiere, según fenotipo, intervención intensiva de estilo de vida, control del peso,
terapias basadas en GLP-1/GIP-GLP-1, SGLT2i, bloqueo SRAA, finerenona, manejo lipídico, HTA, IC, FA, MASLD y apnea del sueño.
            """
        )
        st.info(
            "PREVENT no se calcula internamente en esta versión: se ingresa el valor obtenido por calculadora validada. "
            "Esto evita usar coeficientes no verificados y mantiene la app segura para uso clínico operativo."
        )
        st.markdown("### Próximas ampliaciones sugeridas")
        st.markdown(
            """
1. Integrar cálculo PREVENT oficial si se dispone de coeficientes validados.
2. Incorporar objetivos LDL/no-HDL/ApoB por perfil de riesgo y guías locales.
3. Agregar módulo farmacológico con contraindicaciones, dosis orientativas, interacciones y alertas por eGFR/potasio.
4. Incorporar login, historial por usuario, firma digital e informe PDF institucional.
            """
        )

    st.markdown(f"<div class='footer'>© {datetime.now().year} · {AUTHOR}. Herramienta de apoyo a la decisión clínica; no reemplaza criterio médico ni guías locales.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
