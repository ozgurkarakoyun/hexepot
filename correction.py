"""
Hexapod Fiksatör — Düzeltme Kinematiği
Taylor Spatial Frame — Güncellenmiş Strut Topolojisi

Proksimal Tab bağlantıları:
  Tab 1: S1 (345°) + S2 (15°)
  Tab 2: S3 (105°) + S4 (135°)
  Tab 3: S5 (225°) + S6 (255°)

Distal Tab bağlantıları:
  Tab 1: S6 (45°) + S1 (75°)
  Tab 2: S2 (165°) + S3 (195°)
  Tab 3: S4 (285°) + S5 (315°)
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ─── Sabitler ──────────────────────────────────────────────────
RING_WIDTH  = 15
HOLE_DIAM   = 7
RING_HEIGHT = 8

# Proksimal delik açıları (Tab1: 345°,15° | Tab2: 105°,135° | Tab3: 225°,255°)
PROX_ANGLES = [345, 15, 105, 135, 225, 255]

# Distal delik açıları (Tab1: 45°,75° | Tab2: 165°,195° | Tab3: 285°,315°)
DIST_ANGLES = [45, 75, 165, 195, 285, 315]

# Strut bağlantıları [proks_idx, dist_idx]
# S1: P0(345°)→D1(75°)  S2: P1(15°)→D2(165°)  S3: P2(105°)→D3(195°)
# S4: P3(135°)→D4(285°) S5: P4(225°)→D5(315°)  S6: P5(255°)→D0(45°)
STRUT_CONN  = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]

PAIR_LABELS = ["A","A","B","B","C","C"]  # Proksimal Tab grupları

# Tab tanımları
PROX_TABS = [
    {"name":"Tab 1","struts":[1,2],"hole_idx":[0,1],"center_deg":0,  "start":345,"end":15},
    {"name":"Tab 2","struts":[3,4],"hole_idx":[2,3],"center_deg":120,"start":105,"end":135},
    {"name":"Tab 3","struts":[5,6],"hole_idx":[4,5],"center_deg":240,"start":225,"end":255},
]
DIST_TABS = [
    {"name":"Tab 1","struts":[6,1],"hole_idx":[0,1],"center_deg":60, "start":45, "end":75},
    {"name":"Tab 2","struts":[2,3],"hole_idx":[2,3],"center_deg":180,"start":165,"end":195},
    {"name":"Tab 3","struts":[4,5],"hole_idx":[4,5],"center_deg":300,"start":285,"end":315},
]

# Distal halka deliği → strut haritası (STRUT_CONN'dan türetilir)
# dist_hole[0]=S6, dist_hole[1]=S1, dist_hole[2]=S2, dist_hole[3]=S3, dist_hole[4]=S4, dist_hole[5]=S5
DIST_HOLE_STRUT = [None] * 6
for _si, (_pi, _di) in enumerate(STRUT_CONN):
    DIST_HOLE_STRUT[_di] = _si + 1  # strut numarası (1-indexed)


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


# ─── Klinik deformite rotasyonu ────────────────────────────────
# Koordinat sistemi: Z = distal, X = anterior, Y = medial (sağ tibia)
# +koronal = varus, +sagittal = prokürvatüm, +aksiyel = internal rotasyon
def deformity_rotation(coronal_deg, sagittal_deg, axial_deg):
    return Rz(axial_deg) @ Ry(-sagittal_deg) @ Rx(-coronal_deg)


# ─── Ters kinematik ────────────────────────────────────────────
def calc_strut_lengths(ring_diam, R_dist, t_dist):
    """6 strut uzunluğunu hesapla. Proksimal halka orijinde, Z ekseni distale."""
    mid_r = ring_diam / 2 + RING_WIDTH / 2
    prox_pts = np.array([
        [mid_r * math.cos(math.radians(a)), mid_r * math.sin(math.radians(a)), 0.0]
        for a in PROX_ANGLES
    ])
    dist_local = np.array([
        [mid_r * math.cos(math.radians(a)), mid_r * math.sin(math.radians(a)), 0.0]
        for a in DIST_ANGLES
    ])
    dist_world = (R_dist @ dist_local.T).T + t_dist
    return [round(float(np.linalg.norm(dist_world[di] - prox_pts[pi])), 3)
            for pi, di in STRUT_CONN]


# ─── Pose hesaplayıcı ──────────────────────────────────────────
def compute_poses(separation, cora_dist, coronal, sagittal, axial,
                  ap_trans, ml_trans, length_mm):
    R_neu = np.eye(3)
    t_neu = np.array([0.0, 0.0, float(separation)])
    R_def = deformity_rotation(coronal, sagittal, axial)
    cora  = np.array([0.0, 0.0, float(cora_dist)])
    v0    = np.array([0.0, 0.0, float(separation - cora_dist)])
    t_def = cora + R_def @ v0 + np.array([float(ap_trans), float(ml_trans), float(length_mm)])
    return (R_def, t_def), (R_neu, t_neu)


# ─── SLERP rotasyon interpolasyonu ────────────────────────────
def slerp_rotation(R_start, R_end, alpha):
    r_s = Rotation.from_matrix(R_start)
    r_e = Rotation.from_matrix(R_end)
    return Slerp([0,1], Rotation.concatenate([r_s, r_e]))([alpha])[0].as_matrix()


# ─── Ana düzeltme hesabı ───────────────────────────────────────
def compute_correction_program(
    ring_diam, separation, cora_dist,
    coronal, sagittal, axial,
    ap_trans, ml_trans, length_mm,
    latency_days, correction_days,
    initial_lengths=None,   # Kullanıcı girişi (cerrahi sonrası ölçülen uzunluklar)
):
    """
    Düzeltme programını hesapla.

    initial_lengths: 6 elemanlı liste (mm). Verilmezse deformiteden hesaplanır.
    Hedef uzunluklar her zaman kinematik modelden hesaplanır (nötr pozisyon).

    Returns:
        computed_initial  : kinematik modelden hesaplanan başlangıç uzunlukları
        used_initial      : gerçekte kullanılan başlangıç uzunlukları
        target_lengths    : hedef (düzeltilmiş) strut uzunlukları
        schedule          : günlük takvim
        summary           : özet
    """
    (R_def, t_def), (R_neu, t_neu) = compute_poses(
        separation, cora_dist, coronal, sagittal, axial, ap_trans, ml_trans, length_mm
    )
    computed_initial = calc_strut_lengths(ring_diam, R_def, t_def)
    target_lengths   = calc_strut_lengths(ring_diam, R_neu, t_neu)

    # Başlangıç: kullanıcı girişi veya hesaplanan
    start = [round(float(x), 3) for x in initial_lengths] if initial_lengths else computed_initial

    total_days = latency_days + correction_days
    schedule   = []
    prev = start[:]

    for day in range(total_days + 1):
        if day < latency_days:
            alpha = 0.0
        elif correction_days == 0:
            alpha = 1.0
        else:
            alpha = min((day - latency_days) / correction_days, 1.0)

        # Doğrusal strut uzunluğu interpolasyonu
        current = [round(start[i] + alpha * (target_lengths[i] - start[i]), 3)
                   for i in range(6)]

        changes    = [0.0]*6 if day==0 else [round(current[i]-prev[i],3) for i in range(6)]
        cumulative = [round(current[i]-start[i],3) for i in range(6)]
        phase = "Latent" if day<latency_days else ("Düzeltme" if alpha<1.0 else "Tamamlandı")

        schedule.append({
            "day": day, "lengths": current, "changes": changes,
            "cumulative": cumulative, "alpha": round(alpha,4), "phase": phase,
        })
        prev = current[:]

    max_daily = [max((abs(e["changes"][i]) for e in schedule[1:]), default=0.0) for i in range(6)]

    return {
        "computed_initial": computed_initial,
        "used_initial":     start,
        "target_lengths":   target_lengths,
        "schedule":         schedule,
        "total_days":       total_days,
        "summary": {
            "ring_diam":ring_diam, "outer_diam":ring_diam+RING_WIDTH*2,
            "separation":separation, "mid_r":ring_diam/2+RING_WIDTH/2,
            "cora_dist":cora_dist, "coronal":coronal, "sagittal":sagittal,
            "axial":axial, "ap_trans":ap_trans, "ml_trans":ml_trans,
            "length_mm":length_mm, "latency_days":latency_days,
            "correction_days":correction_days, "total_days":total_days,
            "max_daily_changes":max_daily,
            "input_mode":"manual" if initial_lengths else "computed",
        },
    }


# ─── CSV üretici ──────────────────────────────────────────────
def schedule_to_csv_rows(result):
    rows = [["Gün","Faz","İlerleme%"] +
            [f"S{i+1} Uzunluk(mm)" for i in range(6)] +
            [f"S{i+1} Günlük Δ(mm)" for i in range(6)] +
            [f"S{i+1} Kümülatif(mm)" for i in range(6)]]
    for e in result["schedule"]:
        row = [e["day"], e["phase"], f"{e['alpha']*100:.0f}%"]
        row += e["lengths"] + e["changes"] + e["cumulative"]
        rows.append(row)
    return rows
