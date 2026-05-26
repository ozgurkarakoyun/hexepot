"""
Hexapod Fiksator — Duzeltme Kinematigi
Taylor Spatial Frame / hexapod prototip hesap motoru

Klinik not:
Bu modul cerrahi planlama yardimcisidir. Nihai recete; klinik muayene,
radyolojik olcum, implant/strut mekanik limitleri ve cerrah karari ile dogrulanmalidir.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ─── Sabitler ──────────────────────────────────────────────────
RING_WIDTH  = 15
HOLE_DIAM   = 7
RING_HEIGHT = 8
RING_SIZES  = [120, 150, 180, 210]

DEFAULT_STRUT_MIN = 50.0
DEFAULT_STRUT_MAX = 500.0
DEFAULT_MAX_DAILY_CHANGE = 2.0  # mm/gun; uyarı eşiği, klinik limit olarak doğrulanmalıdır

# Matematik koordinatı: 0° = anterior = +X, 90° = medial = +Y (sağ ekstremite konvansiyonu)
PROX_ANGLES = [345, 15, 105, 135, 225, 255]
DIST_ANGLES = [45, 75, 165, 195, 285, 315]

# STRUT_CONN: [proksimal delik indeksi, distal delik indeksi] — S1..S6
STRUT_CONN  = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
PAIR_LABELS = ["A","A","B","B","C","C"]

PROX_TABS = [
    {"name":"Tab 1","label":"Anterior",       "struts":[1,2],"hole_idx":[0,1],"center_deg":0,  "start":345,"end":15},
    {"name":"Tab 2","label":"Postero-Medial", "struts":[3,4],"hole_idx":[2,3],"center_deg":120,"start":105,"end":135},
    {"name":"Tab 3","label":"Postero-Lateral","struts":[5,6],"hole_idx":[4,5],"center_deg":240,"start":225,"end":255},
]
DIST_TABS = [
    {"name":"Tab 1","label":"Antero-Medial",  "struts":[6,1],"hole_idx":[0,1],"center_deg":60, "start":45, "end":75},
    {"name":"Tab 2","label":"Posterior",      "struts":[2,3],"hole_idx":[2,3],"center_deg":180,"start":165,"end":195},
    {"name":"Tab 3","label":"Antero-Lateral", "struts":[4,5],"hole_idx":[4,5],"center_deg":300,"start":285,"end":315},
]

DIST_HOLE_STRUT = [None] * 6
for _si, (_pi, _di) in enumerate(STRUT_CONN):
    DIST_HOLE_STRUT[_di] = _si + 1

# ─── Yardimci ve validasyon ────────────────────────────────────
def Rx(deg):
    t = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(t),-math.sin(t)],[0,math.sin(t),math.cos(t)]])

def Ry(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t),0,math.sin(t)],[0,1,0],[-math.sin(t),0,math.cos(t)]])

def Rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1]])

def _as_float(name, value):
    try:
        v = float(value)
    except Exception:
        raise ValueError(f"{name} sayisal olmalidir.")
    if not math.isfinite(v):
        raise ValueError(f"{name} gecerli/sonlu bir sayi olmalidir.")
    return v

def _as_int(name, value):
    try:
        return int(value)
    except Exception:
        raise ValueError(f"{name} tam sayi olmalidir.")

def _check_range(name, v, lo, hi):
    if v < lo or v > hi:
        raise ValueError(f"{name} {lo} ile {hi} arasinda olmalidir. Girilen: {v}")

def normalize_inputs(
    ring_diam, separation, cora_dist, reference_ring,
    coronal, sagittal, axial, ap_trans, ml_trans, length_mm,
    latency_days, correction_days, side='right', bone='tibia', segment='distal',
    strut_min=DEFAULT_STRUT_MIN, strut_max=DEFAULT_STRUT_MAX,
    max_daily_change=DEFAULT_MAX_DAILY_CHANGE, initial_lengths=None
):
    ring_diam = _as_int("ring_diam", ring_diam)
    if ring_diam not in RING_SIZES:
        raise ValueError(f"Halka boyutu desteklenmiyor: {ring_diam}. Desteklenen: {RING_SIZES}")

    separation = _as_float("separation", separation)
    cora_dist  = _as_float("cora_dist", cora_dist)
    _check_range("Halka araligi", separation, 40.0, 250.0)
    _check_range("CORA mesafesi", cora_dist, 0.0, separation)

    reference_ring = str(reference_ring or 'proximal').lower()
    if reference_ring not in {'proximal', 'distal'}:
        raise ValueError("reference_ring 'proximal' veya 'distal' olmalidir.")

    side = str(side or 'right').lower()
    if side not in {'right', 'left'}:
        raise ValueError("side 'right' veya 'left' olmalidir.")

    bone = str(bone or 'tibia').lower()
    if bone not in {'tibia', 'femur'}:
        raise ValueError("bone 'tibia' veya 'femur' olmalidir.")

    segment = str(segment or 'distal').lower()
    if segment not in {'proximal', 'distal'}:
        raise ValueError("segment 'proximal' veya 'distal' olmalidir.")

    coronal  = _as_float("coronal", coronal)
    sagittal = _as_float("sagittal", sagittal)
    axial    = _as_float("axial", axial)
    for name, v in [("Koronal aci", coronal), ("Sagittal aci", sagittal), ("Aksiyel aci", axial)]:
        _check_range(name, v, -90.0, 90.0)

    ap_trans  = _as_float("ap_trans", ap_trans)
    ml_trans  = _as_float("ml_trans", ml_trans)
    length_mm = _as_float("length_mm", length_mm)
    for name, v in [("AP translasyon", ap_trans), ("ML translasyon", ml_trans), ("Boy/uzatma", length_mm)]:
        _check_range(name, v, -100.0, 100.0)

    latency_days    = _as_int("latency_days", latency_days)
    correction_days = _as_int("correction_days", correction_days)
    _check_range("Latent sure", latency_days, 0, 60)
    _check_range("Duzeltme suresi", correction_days, 0, 180)

    strut_min = _as_float("strut_min", strut_min)
    strut_max = _as_float("strut_max", strut_max)
    max_daily_change = _as_float("max_daily_change", max_daily_change)
    if strut_min <= 0 or strut_max <= strut_min:
        raise ValueError("Strut minimum/maksimum limitleri hatali.")
    _check_range("Gunluk maksimum degisim", max_daily_change, 0.1, 20.0)

    if initial_lengths:
        if len(initial_lengths) != 6:
            raise ValueError("initial_lengths tam 6 strut uzunlugu icermelidir.")
        initial_lengths = [_as_float(f"S{i+1} baslangic", x) for i, x in enumerate(initial_lengths)]
        for i, x in enumerate(initial_lengths):
            if x < strut_min or x > strut_max:
                raise ValueError(f"S{i+1} baslangic uzunlugu strut limiti disinda: {x} mm")

    return dict(
        ring_diam=ring_diam, separation=separation, cora_dist=cora_dist,
        reference_ring=reference_ring, side=side, bone=bone, segment=segment,
        coronal=coronal, sagittal=sagittal, axial=axial,
        ap_trans=ap_trans, ml_trans=ml_trans, length_mm=length_mm,
        latency_days=latency_days, correction_days=correction_days,
        strut_min=strut_min, strut_max=strut_max, max_daily_change=max_daily_change,
        initial_lengths=initial_lengths,
    )

# ─── Klinik transform ──────────────────────────────────────────
def apply_side_convention(ml_trans, side):
    """Sol ekstremitede medial-lateral aks klinik olarak aynalanir."""
    return -float(ml_trans) if side == 'left' else float(ml_trans)

def deformity_rotation(coronal, sagittal, axial, side='right', bone='tibia', segment='distal'):
    """
    Klinik acilari rotasyon matrisine cevirir.
    Varsayilan konvansiyon: sag tibia, distal segment hareketli.
    Sol taraf icin koronal/aksiyel yonler aynalanir.
    Femur icin su an ayni matematiksel model kullanilir; cikti summary'de uyarilir.
    """
    c = -float(coronal) if side == 'left' else float(coronal)
    a = -float(axial)   if side == 'left' else float(axial)
    s = float(sagittal)
    R = Rz(a) @ Ry(-s) @ Rx(-c)
    if segment == 'proximal':
        # Hareketli segment proksimal kabul edildiginde distal referansa gore ters transform yaklasimi
        R = R.T
    return R

# ─── Ters kinematik ────────────────────────────────────────────
def calc_strut_lengths(ring_diam, R_dist, t_dist):
    mid_r = ring_diam / 2 + RING_WIDTH / 2
    prox = np.array([[mid_r*math.cos(math.radians(a)), mid_r*math.sin(math.radians(a)), 0.0] for a in PROX_ANGLES])
    dist_local = np.array([[mid_r*math.cos(math.radians(a)), mid_r*math.sin(math.radians(a)), 0.0] for a in DIST_ANGLES])
    dist_world = (R_dist @ dist_local.T).T + t_dist
    return [round(float(np.linalg.norm(dist_world[di] - prox[pi])), 3) for pi, di in STRUT_CONN]

# ─── Montaj + CORA konumu ─────────────────────────────────────
def resolve_cora_position(separation, cora_dist, reference_ring='proximal'):
    if reference_ring == 'proximal':
        return float(cora_dist)
    return float(separation) - float(cora_dist)

def compute_poses(separation, cora_dist, coronal, sagittal, axial,
                  ap_trans, ml_trans, length_mm, reference_ring='proximal',
                  side='right', bone='tibia', segment='distal'):
    R_neu = np.eye(3)
    t_neu = np.array([0.0, 0.0, float(separation)])
    R_def = deformity_rotation(coronal, sagittal, axial, side, bone, segment)

    cora_z = resolve_cora_position(separation, cora_dist, reference_ring)
    cora = np.array([0.0, 0.0, cora_z])
    v0 = np.array([0.0, 0.0, float(separation) - cora_z])
    ml_model = apply_side_convention(ml_trans, side)
    t_def = cora + R_def @ v0 + np.array([float(ap_trans), ml_model, float(length_mm)])
    return (R_def, t_def), (R_neu, t_neu)

# ─── SLERP ────────────────────────────────────────────────────
def slerp_R(R0, R1, t):
    r0 = Rotation.from_matrix(R0)
    r1 = Rotation.from_matrix(R1)
    return Slerp([0, 1], Rotation.concatenate([r0, r1]))([t])[0].as_matrix()

# ─── Ana duzeltme hesabi ───────────────────────────────────────
def compute_correction_program(
    ring_diam, separation, cora_dist, reference_ring='proximal',
    coronal=0, sagittal=0, axial=0, ap_trans=0, ml_trans=0, length_mm=0,
    latency_days=7, correction_days=30, initial_lengths=None,
    side='right', bone='tibia', segment='distal',
    strut_min=DEFAULT_STRUT_MIN, strut_max=DEFAULT_STRUT_MAX,
    max_daily_change=DEFAULT_MAX_DAILY_CHANGE,
):
    """
    Pose-based gunluk duzeltme programi.
    Her gun icin rotasyon SLERP, translasyon lineer interpolasyon ile ara poz hesaplanir;
    strut uzunluklari bu ara pozdan yeniden turetilir.
    """
    p = normalize_inputs(
        ring_diam, separation, cora_dist, reference_ring,
        coronal, sagittal, axial, ap_trans, ml_trans, length_mm,
        latency_days, correction_days, side, bone, segment,
        strut_min, strut_max, max_daily_change, initial_lengths
    )

    (R_def, t_def), (R_neu, t_neu) = compute_poses(
        p['separation'], p['cora_dist'], p['coronal'], p['sagittal'], p['axial'],
        p['ap_trans'], p['ml_trans'], p['length_mm'], p['reference_ring'],
        p['side'], p['bone'], p['segment']
    )
    computed_initial = calc_strut_lengths(p['ring_diam'], R_def, t_def)
    target_lengths = calc_strut_lengths(p['ring_diam'], R_neu, t_neu)
    start = [round(float(x), 3) for x in p['initial_lengths']] if p['initial_lengths'] else computed_initial

    total_days = p['latency_days'] + p['correction_days']
    schedule = []
    prev = start[:]
    warnings = []

    if p['bone'] == 'femur':
        warnings.append("Femur secildi: model genel koordinat konvansiyonu kullanir; klinik eksenler cerrah tarafindan dogrulanmalidir.")
    if p['segment'] == 'proximal':
        warnings.append("Proksimal segment hareketli secildi: transform ters yon yaklasimi ile hesaplanir; radyografik dogrulama gerekir.")

    for day in range(total_days + 1):
        if day < p['latency_days']:
            alpha = 0.0
            current = start[:]
        else:
            alpha = 1.0 if p['correction_days'] == 0 else min((day - p['latency_days']) / p['correction_days'], 1.0)
            if p['initial_lengths']:
                # Manuel baslangic girilmisse mekanik recete fiili start -> hedef arasinda ilerler.
                current = [round(start[i] + alpha * (target_lengths[i] - start[i]), 3) for i in range(6)]
            else:
                # Pose-based: deformite pozundan notr poza gunluk ara poz.
                R_cur = slerp_R(R_def, R_neu, alpha)
                t_cur = (1.0 - alpha) * t_def + alpha * t_neu
                current = calc_strut_lengths(p['ring_diam'], R_cur, t_cur)

        changes = [0.0] * 6 if day == 0 else [round(current[i] - prev[i], 3) for i in range(6)]
        cumulative = [round(current[i] - start[i], 3) for i in range(6)]
        phase = "Latent" if day < p['latency_days'] else ("Duzeltme" if alpha < 1.0 else "Tamamlandi")

        day_warnings = []
        for i, L in enumerate(current):
            if L < p['strut_min'] or L > p['strut_max']:
                msg = f"Gun {day}: S{i+1} strut uzunlugu limit disinda ({L} mm; limit {p['strut_min']}-{p['strut_max']} mm)."
                day_warnings.append(msg)
                warnings.append(msg)
            if abs(changes[i]) > p['max_daily_change']:
                msg = f"Gun {day}: S{i+1} gunluk degisim {changes[i]} mm; esik {p['max_daily_change']} mm/gun."
                day_warnings.append(msg)
                warnings.append(msg)

        schedule.append({
            "day": day, "lengths": current, "changes": changes,
            "cumulative": cumulative, "alpha": round(alpha, 4), "phase": phase,
            "warnings": day_warnings,
        })
        prev = current[:]

    max_daily = [max((abs(e['changes'][i]) for e in schedule[1:]), default=0.0) for i in range(6)]
    cora_z = resolve_cora_position(p['separation'], p['cora_dist'], p['reference_ring'])
    unique_warnings = list(dict.fromkeys(warnings))

    return {
        "computed_initial": computed_initial,
        "used_initial": start,
        "target_lengths": target_lengths,
        "schedule": schedule,
        "total_days": total_days,
        "warnings": unique_warnings,
        "disclaimer": "Cerrahi planlama yardimcisidir. Nihai klinik karar ve strut recetesi cerrah tarafindan radyolojik/klinik olarak dogrulanmalidir.",
        "summary": {
            "ring_diam": p['ring_diam'], "outer_diam": p['ring_diam'] + RING_WIDTH * 2,
            "separation": p['separation'], "mid_r": p['ring_diam']/2 + RING_WIDTH/2,
            "reference_ring": p['reference_ring'], "cora_dist": p['cora_dist'], "cora_z_from_prox": cora_z,
            "side": p['side'], "bone": p['bone'], "segment": p['segment'],
            "coronal": p['coronal'], "sagittal": p['sagittal'], "axial": p['axial'],
            "ap_trans": p['ap_trans'], "ml_trans": p['ml_trans'], "length_mm": p['length_mm'],
            "latency_days": p['latency_days'], "correction_days": p['correction_days'], "total_days": total_days,
            "strut_min": p['strut_min'], "strut_max": p['strut_max'], "max_daily_change": p['max_daily_change'],
            "max_daily_changes": max_daily,
            "input_mode": "manual" if p['initial_lengths'] else "pose_based",
            "schedule_method": "manual_start_linear_to_target" if p['initial_lengths'] else "pose_based_slerp_translation",
        },
    }

# ─── CSV ──────────────────────────────────────────────────────
def schedule_to_csv_rows(result):
    rows = [["Gün", "Faz", "İlerleme%"]
            + [f"S{i+1} (mm)" for i in range(6)]
            + [f"S{i+1} Δ (mm)" for i in range(6)]
            + [f"S{i+1} Kümülatif (mm)" for i in range(6)]
            + ["Uyarılar"]]
    for e in result["schedule"]:
        rows.append([e["day"], e["phase"], f"{e['alpha']*100:.0f}%"]
                    + e["lengths"] + e["changes"] + e["cumulative"]
                    + [" | ".join(e.get("warnings", []))])
    return rows
