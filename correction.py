"""
Hexapod Fiksatör — Düzeltme Kinematiği Modülü
Taylor Spatial Frame inverse/forward kinematik + düzeltme takvimi hesabı
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ─── Sabitler ──────────────────────────────────────────────────
RING_WIDTH  = 15
HOLE_DIAM   = 7
PROX_ANGLES = [345, 15, 105, 135, 225, 255]   # proksimal bağlantı açıları
DIST_ANGLES = [45,  75, 165, 195, 285, 315]    # distal bağlantı açıları
STRUT_CONN  = [[0,1],[1,0],[2,3],[3,2],[4,5],[5,4]]  # [proks_idx, dist_idx]
PAIR_LABELS = ["A","A","B","B","C","C"]


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


# ─── Klinik konvansiyonlar ─────────────────────────────────────
# Koordinat sistemi (sağ tibia, önden bakış):
#   Z: proksimalden distale (+Z = distal)
#   X: anterior (+)
#   Y: medial (+)
#
# İşaret kuralları:
#   Koronal   +: Varus   (distal mediale kayar)
#   Sagittal  +: Prokürvatüm / Apex anterior  (distal posteriora kayar)
#   Aksiyel   +: İnternal rotasyon
#   AP trans  +: Anterior translasyon
#   ML trans  +: Medial translasyon
#   Boy       +: Uzatma (distraksiyon)

def deformity_rotation(coronal_deg, sagittal_deg, axial_deg):
    """
    Klinik deformite açılarından rotasyon matrisi üret.
    Rx(-koronal): +koronal → Y pozitif → medial → varus ✓
    Ry(-sagittal): +sagittal → X negatif → posterior → prokürvatüm ✓
    Rz(aksiyel):  +aksiyel → internal rotasyon ✓
    """
    return Rz(axial_deg) @ Ry(-sagittal_deg) @ Rx(-coronal_deg)


# ─── Ters kinematik (strut uzunluğu hesabı) ────────────────────
def calc_strut_lengths(ring_diam, R_dist, t_dist):
    """
    6 strutun uzunluğunu hesapla.

    Paramlar:
        ring_diam : Halka iç çapı (mm)
        R_dist    : Distal halkanın rotasyon matrisi (3×3)
        t_dist    : Distal halka merkezi konumu (3,) — proksimal halka origin
    
    Proksimal halka: orijinde (0,0,0), kimlik rotasyonu.
    Z ekseni distal yönde pozitif.
    """
    mid_r = ring_diam / 2 + RING_WIDTH / 2

    # Proksimal bağlantı noktaları (dünya koordinatı, sabit)
    prox_pts = np.array([
        [mid_r * math.cos(math.radians(a)),
         mid_r * math.sin(math.radians(a)), 0.0]
        for a in PROX_ANGLES
    ])

    # Distal bağlantı noktaları (yerel çerçeve)
    dist_local = np.array([
        [mid_r * math.cos(math.radians(a)),
         mid_r * math.sin(math.radians(a)), 0.0]
        for a in DIST_ANGLES
    ])

    # Dünya koordinatına dönüştür
    dist_world = (R_dist @ dist_local.T).T + t_dist

    lengths = []
    for pi, di in STRUT_CONN:
        L = float(np.linalg.norm(dist_world[di] - prox_pts[pi]))
        lengths.append(round(L, 3))
    return lengths


# ─── Pose hesaplayıcı ─────────────────────────────────────────
def compute_poses(separation, cora_dist,
                  coronal, sagittal, axial,
                  ap_trans, ml_trans, length_mm):
    """
    Deformiteli ve düzeltilmiş (nötr) distal halka pozisyonlarını döndür.
    
    Returns:
        (R_def, t_def) — deformiteli pozisyon
        (R_neu, t_neu) — nötr (düzeltilmiş) pozisyon
    """
    # Nötr: halkalar koaksiyel, paralel
    R_neu = np.eye(3)
    t_neu = np.array([0.0, 0.0, float(separation)])

    # Deformite rotasyonu
    R_def = deformity_rotation(coronal, sagittal, axial)

    # CORA pozisyonu
    cora = np.array([0.0, 0.0, float(cora_dist)])

    # CORA'dan nötr distal halkaya vektör
    v0 = np.array([0.0, 0.0, float(separation - cora_dist)])

    # Bu vektörü deformite ile döndür
    v_def = R_def @ v0

    # Ek translasyon + boy farkı
    t_extra = np.array([float(ap_trans), float(ml_trans), float(length_mm)])

    # Deformiteli distal halka merkezi
    t_def = cora + v_def + t_extra

    return (R_def, t_def), (R_neu, t_neu)


# ─── Rotasyon interpolasyonu (SLERP) ──────────────────────────
def slerp_rotation(R_start, R_end, alpha):
    """SLERP: alpha=0 → R_start, alpha=1 → R_end"""
    r_start = Rotation.from_matrix(R_start)
    r_end   = Rotation.from_matrix(R_end)
    slerp   = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))
    return slerp([alpha])[0].as_matrix()


# ─── Ana düzeltme hesabı ───────────────────────────────────────
def compute_correction_program(
    ring_diam, separation,
    cora_dist,
    coronal, sagittal, axial,
    ap_trans, ml_trans, length_mm,
    latency_days, correction_days
):
    """
    Tam düzeltme programını hesapla.

    Returns dict:
        neutral_lengths   : nötr strut uzunlukları (hedef)
        deformed_lengths  : deformiteli strut uzunlukları (başlangıç)
        schedule          : günlük liste [{day, lengths, changes, alpha, cumulative_change}]
        total_days        : toplam süre
        summary           : özet parametreler
    """
    (R_def, t_def), (R_neu, t_neu) = compute_poses(
        separation, cora_dist,
        coronal, sagittal, axial,
        ap_trans, ml_trans, length_mm
    )

    neutral_lengths  = calc_strut_lengths(ring_diam, R_neu, t_neu)
    deformed_lengths = calc_strut_lengths(ring_diam, R_def, t_def)

    total_days = latency_days + correction_days
    schedule   = []
    prev_lengths = deformed_lengths[:]

    for day in range(total_days + 1):
        if day < latency_days:
            alpha = 0.0
        elif correction_days == 0:
            alpha = 1.0
        else:
            alpha = min((day - latency_days) / correction_days, 1.0)

        R_cur = slerp_rotation(R_def, R_neu, alpha)
        t_cur = t_def + alpha * (t_neu - t_def)

        current_lengths = calc_strut_lengths(ring_diam, R_cur, t_cur)

        daily_changes = (
            [0.0] * 6 if day == 0
            else [round(current_lengths[i] - prev_lengths[i], 3) for i in range(6)]
        )
        cumulative = [round(current_lengths[i] - deformed_lengths[i], 3) for i in range(6)]

        schedule.append({
            "day":        day,
            "lengths":    current_lengths,
            "changes":    daily_changes,
            "cumulative": cumulative,
            "alpha":      round(alpha, 4),
            "phase":      "Latent" if day < latency_days else ("Düzeltme" if alpha < 1.0 else "Tamamlandı"),
        })
        prev_lengths = current_lengths[:]

    # Günlük maksimum strut değişimleri
    max_daily = [0.0] * 6
    for entry in schedule[1:]:
        for i in range(6):
            if abs(entry["changes"][i]) > abs(max_daily[i]):
                max_daily[i] = entry["changes"][i]

    summary = {
        "ring_diam":         ring_diam,
        "outer_diam":        ring_diam + RING_WIDTH * 2,
        "separation":        separation,
        "mid_r":             ring_diam / 2 + RING_WIDTH / 2,
        "cora_dist":         cora_dist,
        "coronal":           coronal,
        "sagittal":          sagittal,
        "axial":             axial,
        "ap_trans":          ap_trans,
        "ml_trans":          ml_trans,
        "length_mm":         length_mm,
        "latency_days":      latency_days,
        "correction_days":   correction_days,
        "total_days":        total_days,
        "max_daily_changes": max_daily,
        "t_def":             t_def.tolist(),
        "t_neu":             t_neu.tolist(),
    }

    return {
        "neutral_lengths":  neutral_lengths,
        "deformed_lengths": deformed_lengths,
        "schedule":         schedule,
        "total_days":       total_days,
        "summary":          summary,
    }


# ─── CSV üretici ──────────────────────────────────────────────
def schedule_to_csv_rows(result):
    """Düzeltme tablosunu CSV satırlarına dönüştür."""
    rows = []
    header = ["Gün", "Faz", "İlerleme%"]
    for i in range(1, 7):
        header += [f"S{i} Uzunluk(mm)", f"S{i} Günlük(mm)", f"S{i} Kümülatif(mm)"]
    rows.append(header)
    for e in result["schedule"]:
        row = [e["day"], e["phase"], f"{e['alpha']*100:.1f}%"]
        for i in range(6):
            row += [e["lengths"][i], e["changes"][i], e["cumulative"][i]]
        rows.append(row)
    return rows
