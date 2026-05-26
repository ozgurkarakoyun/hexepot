"""
Hexapod Fiksatör — Düzeltme Kinematiği
Taylor Spatial Frame — TSF Standart

Klinik koordinat sistemi (sağ tibia):
  Z: proksimalden distale (+Z = distal)
  X: anterior (+)
  Y: medial (+)

Tab bağlantıları:
  Proks Tab1 (anterior, 0°):   S1(345°) + S2(15°)
  Proks Tab2 (postero-med, 120°): S3(105°) + S4(135°)
  Proks Tab3 (postero-lat, 240°): S5(225°) + S6(255°)

  Dist Tab1 (anteromed, 60°):  S6(45°)  + S1(75°)
  Dist Tab2 (posterior, 180°): S2(165°) + S3(195°)
  Dist Tab3 (anterolat, 300°): S4(285°) + S5(315°)

STRUT_CONN: S1[P0→D1] S2[P1→D2] S3[P2→D3] S4[P3→D4] S5[P4→D5] S6[P5→D0]
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ─── Sabitler ──────────────────────────────────────────────────
RING_WIDTH  = 15
HOLE_DIAM   = 7
RING_HEIGHT = 8

# Proksimal delik açıları (matematik koordinatı: 0°=anterior=+X)
PROX_ANGLES = [345, 15, 105, 135, 225, 255]
# Distal delik açıları
DIST_ANGLES = [45, 75, 165, 195, 285, 315]

# STRUT_CONN: [proks_idx, dist_idx] — S1..S6
STRUT_CONN  = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
PAIR_LABELS = ["A","A","B","B","C","C"]

# Tab tanımları (klinik açıklamalar)
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

# Distal delik → strut no haritası
DIST_HOLE_STRUT = [None] * 6
for _si, (_pi, _di) in enumerate(STRUT_CONN):
    DIST_HOLE_STRUT[_di] = _si + 1


# ─── Rotasyon yardımcıları ─────────────────────────────────────
def Rx(deg):
    t = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(t),-math.sin(t)],[0,math.sin(t),math.cos(t)]])
def Ry(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t),0,math.sin(t)],[0,1,0],[-math.sin(t),0,math.cos(t)]])
def Rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1]])

# İşaret kuralı:
#   +koronal  = varus        (distal mediale kayar)
#   +sagittal = prokürvatüm  (apex anterior, distal posteriora kayar)
#   +aksiyel  = internal rot
def deformity_rotation(coronal, sagittal, axial):
    return Rz(axial) @ Ry(-sagittal) @ Rx(-coronal)


# ─── Ters kinematik ────────────────────────────────────────────
def calc_strut_lengths(ring_diam, R_dist, t_dist):
    mid_r = ring_diam / 2 + RING_WIDTH / 2
    prox  = np.array([[mid_r*math.cos(math.radians(a)),
                        mid_r*math.sin(math.radians(a)), 0.0] for a in PROX_ANGLES])
    dist_local = np.array([[mid_r*math.cos(math.radians(a)),
                             mid_r*math.sin(math.radians(a)), 0.0] for a in DIST_ANGLES])
    dist_world = (R_dist @ dist_local.T).T + t_dist
    return [round(float(np.linalg.norm(dist_world[di] - prox[pi])), 3)
            for pi, di in STRUT_CONN]


# ─── Montaj + CORA konumu ─────────────────────────────────────
def resolve_cora_position(separation, cora_dist, reference_ring='proximal'):
    """
    CORA'nın proksimal halka merkezine göre Z koordinatını hesapla.

    reference_ring='proximal':
        CORA, proksimal halka merkezinden itibaren distal yönde cora_dist mm
    reference_ring='distal':
        CORA, distal halka merkezinden itibaren proksimal yönde cora_dist mm
    """
    if reference_ring == 'proximal':
        return float(cora_dist)
    else:  # distal
        return float(separation) - float(cora_dist)


# ─── Pose hesaplayıcı ──────────────────────────────────────────
def compute_poses(separation, cora_dist, coronal, sagittal, axial,
                  ap_trans, ml_trans, length_mm, reference_ring='proximal'):
    R_neu = np.eye(3)
    t_neu = np.array([0.0, 0.0, float(separation)])
    R_def = deformity_rotation(coronal, sagittal, axial)

    cora_z = resolve_cora_position(separation, cora_dist, reference_ring)
    cora   = np.array([0.0, 0.0, cora_z])
    v0     = np.array([0.0, 0.0, float(separation) - cora_z])
    t_def  = cora + R_def @ v0 + np.array([float(ap_trans), float(ml_trans), float(length_mm)])

    return (R_def, t_def), (R_neu, t_neu)


# ─── SLERP ────────────────────────────────────────────────────
def slerp_R(R0, R1, t):
    r0 = Rotation.from_matrix(R0); r1 = Rotation.from_matrix(R1)
    return Slerp([0,1], Rotation.concatenate([r0,r1]))([t])[0].as_matrix()


# ─── Ana düzeltme hesabı ───────────────────────────────────────
def compute_correction_program(
    ring_diam, separation,
    cora_dist, reference_ring,
    coronal, sagittal, axial,
    ap_trans, ml_trans, length_mm,
    latency_days, correction_days,
    initial_lengths=None,
):
    """
    Düzeltme programı hesabı.

    initial_lengths: 6 elemanlı liste (mm).
        Verilirse → kullanıcı girişi (cerrahi sonrası ölçülen değerler).
        Verilmezse → deformite parametrelerinden kinematik hesaplama.

    target_lengths: Her zaman kinematik modelden → nötr pozisyon.
    Schedule: start → target arası doğrusal interpolasyon.
    """
    (R_def, t_def), (R_neu, t_neu) = compute_poses(
        separation, cora_dist, coronal, sagittal, axial,
        ap_trans, ml_trans, length_mm, reference_ring
    )
    computed_initial = calc_strut_lengths(ring_diam, R_def, t_def)
    target_lengths   = calc_strut_lengths(ring_diam, R_neu, t_neu)
    start = [round(float(x),3) for x in initial_lengths] if initial_lengths else computed_initial

    total_days = latency_days + correction_days
    schedule   = []
    prev       = start[:]

    for day in range(total_days + 1):
        if day < latency_days:
            alpha = 0.0
        elif correction_days == 0:
            alpha = 1.0
        else:
            alpha = min((day - latency_days) / correction_days, 1.0)

        current    = [round(start[i] + alpha*(target_lengths[i]-start[i]), 3) for i in range(6)]
        changes    = [0.0]*6 if day==0 else [round(current[i]-prev[i], 3) for i in range(6)]
        cumulative = [round(current[i]-start[i], 3) for i in range(6)]
        phase      = "Latent" if day<latency_days else ("Düzeltme" if alpha<1.0 else "Tamamlandı")

        schedule.append({"day":day,"lengths":current,"changes":changes,
                         "cumulative":cumulative,"alpha":round(alpha,4),"phase":phase})
        prev = current[:]

    max_daily = [max((abs(e["changes"][i]) for e in schedule[1:]), default=0.0) for i in range(6)]
    cora_z    = resolve_cora_position(separation, cora_dist, reference_ring)

    return {
        "computed_initial": computed_initial,
        "used_initial":     start,
        "target_lengths":   target_lengths,
        "schedule":         schedule,
        "total_days":       total_days,
        "summary": {
            "ring_diam":ring_diam, "outer_diam":ring_diam+RING_WIDTH*2,
            "separation":separation, "mid_r":ring_diam/2+RING_WIDTH/2,
            "reference_ring":reference_ring, "cora_dist":cora_dist,
            "cora_z_from_prox":cora_z,
            "coronal":coronal,"sagittal":sagittal,"axial":axial,
            "ap_trans":ap_trans,"ml_trans":ml_trans,"length_mm":length_mm,
            "latency_days":latency_days,"correction_days":correction_days,
            "total_days":total_days,"max_daily_changes":max_daily,
            "input_mode":"manual" if initial_lengths else "computed",
        },
    }


# ─── CSV ──────────────────────────────────────────────────────
def schedule_to_csv_rows(result):
    rows = [["Gün","Faz","İlerleme%"]
            + [f"S{i+1} (mm)" for i in range(6)]
            + [f"S{i+1} Δ (mm)" for i in range(6)]
            + [f"S{i+1} Kümülatif (mm)" for i in range(6)]]
    for e in result["schedule"]:
        rows.append([e["day"],e["phase"],f"{e['alpha']*100:.0f}%"]
                    + e["lengths"] + e["changes"] + e["cumulative"])
    return rows
