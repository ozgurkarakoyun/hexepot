"""
Hexapod Fiksatör — Flask Web Uygulaması
"""
import os, io, csv, math
import numpy as np
from flask import Flask, render_template, jsonify, request, Response

from correction import (
    compute_correction_program, schedule_to_csv_rows,
    calc_strut_lengths, compute_poses,
    RING_WIDTH, HOLE_DIAM, RING_HEIGHT,
    PROX_ANGLES, DIST_ANGLES, STRUT_CONN,
    PROX_TABS, DIST_TABS, DIST_HOLE_STRUT, PAIR_LABELS
)

app   = Flask(__name__)
SIZES = [120, 150, 180, 210]

# ─── Yardımcı ─────────────────────────────────────────────────
def npfix(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: npfix(v) for k,v in obj.items()}
    if isinstance(obj, list): return [npfix(i) for i in obj]
    return obj

def make_holes(inner_diam, angles):
    mid_r = inner_diam/2 + RING_WIDTH/2
    return [{
        "id": i+1, "pair": PAIR_LABELS[i], "angle": a,
        "x": round(mid_r * math.cos(math.radians(a)), 3),
        "y": round(mid_r * math.sin(math.radians(a)), 3),
        "mid_r": mid_r,
    } for i, a in enumerate(angles)]

# ─── Sayfalar ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── API: Halka geometrisi ────────────────────────────────────
@app.route("/api/ring")
def api_ring():
    d     = int(request.args.get("diam", 150))
    rtype = request.args.get("type", "proximal")
    angles = PROX_ANGLES if rtype=="proximal" else DIST_ANGLES
    tabs   = PROX_TABS   if rtype=="proximal" else DIST_TABS
    holes  = make_holes(d, angles)

    # Hangi deliğin hangi strut'a ait olduğunu ekle
    if rtype == "proximal":
        strut_map = [si+1 for si in range(6)]   # prox hole i → strut i+1
    else:
        strut_map = DIST_HOLE_STRUT              # [6,1,2,3,4,5]

    for i, h in enumerate(holes):
        h["strut_no"] = strut_map[i]

    return jsonify({
        "inner_diam":d, "outer_diam":d+RING_WIDTH*2,
        "ring_width":RING_WIDTH, "ring_height":RING_HEIGHT,
        "mid_r":d/2+RING_WIDTH/2, "hole_diam":HOLE_DIAM,
        "ring_type":rtype, "holes":holes, "tabs":tabs,
        "strut_conn":STRUT_CONN,
        "prox_angles":PROX_ANGLES, "dist_angles":DIST_ANGLES,
    })

# ─── API: Tüm halkalar ───────────────────────────────────────
@app.route("/api/all-rings")
def api_all_rings():
    rtype  = request.args.get("type","proximal")
    angles = PROX_ANGLES if rtype=="proximal" else DIST_ANGLES
    return jsonify([{
        "inner_diam":d, "outer_diam":d+RING_WIDTH*2,
        "mid_r":d/2+RING_WIDTH/2,
        "holes": make_holes(d, angles),
    } for d in SIZES])

# ─── API: Başlangıç uzunluklarını hesapla ────────────────────
@app.route("/api/compute-initial", methods=["POST"])
def api_compute_initial():
    """Deformite parametrelerinden başlangıç strut uzunluklarını hesapla."""
    d = request.json
    try:
        (R_def, t_def), _ = compute_poses(
            float(d["separation"]), float(d["cora_dist"]),
            float(d["coronal"]), float(d["sagittal"]), float(d["axial"]),
            float(d["ap_trans"]), float(d["ml_trans"]), float(d["length_mm"]),
        )
        lengths = calc_strut_lengths(int(d["ring_diam"]), R_def, t_def)
        return jsonify({"initial_lengths": lengths})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── API: Nötr uzunlukları ───────────────────────────────────
@app.route("/api/neutral-lengths")
def api_neutral():
    d   = int(request.args.get("diam", 150))
    sep = float(request.args.get("sep", 120))
    mid_r = d/2 + RING_WIDTH/2
    t_neu = np.array([0.0, 0.0, sep])
    lengths = calc_strut_lengths(d, np.eye(3), t_neu)
    return jsonify({"neutral_lengths": lengths, "mid_r": mid_r})

# ─── API: Düzeltme programı ──────────────────────────────────
@app.route("/api/correction", methods=["POST"])
def api_correction():
    d = request.json
    try:
        result = compute_correction_program(
            ring_diam       = int(d["ring_diam"]),
            separation      = float(d["separation"]),
            cora_dist       = float(d["cora_dist"]),
            coronal         = float(d["coronal"]),
            sagittal        = float(d["sagittal"]),
            axial           = float(d["axial"]),
            ap_trans        = float(d["ap_trans"]),
            ml_trans        = float(d["ml_trans"]),
            length_mm       = float(d["length_mm"]),
            latency_days    = int(d["latency_days"]),
            correction_days = int(d["correction_days"]),
            initial_lengths = d.get("initial_lengths"),  # Opsiyonel
        )
        return jsonify(npfix(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── API: Takvim CSV ─────────────────────────────────────────
@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    d = request.json
    try:
        result = compute_correction_program(
            ring_diam       = int(d["ring_diam"]),
            separation      = float(d["separation"]),
            cora_dist       = float(d["cora_dist"]),
            coronal         = float(d["coronal"]),
            sagittal        = float(d["sagittal"]),
            axial           = float(d["axial"]),
            ap_trans        = float(d["ap_trans"]),
            ml_trans        = float(d["ml_trans"]),
            length_mm       = float(d["length_mm"]),
            latency_days    = int(d["latency_days"]),
            correction_days = int(d["correction_days"]),
            initial_lengths = d.get("initial_lengths"),
        )
        rows = schedule_to_csv_rows(result)
        out  = io.StringIO()
        csv.writer(out).writerows(rows)
        return Response("\ufeff"+out.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":"attachment;filename=hexapod_takvim.csv"})
    except Exception as e:
        return jsonify({"error":str(e)}), 400

# ─── API: Koordinat CSV ──────────────────────────────────────
@app.route("/api/export/coords")
def export_coords():
    rtype  = request.args.get("type","proximal")
    angles = PROX_ANGLES if rtype=="proximal" else DIST_ANGLES
    out    = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Halka İç Çap","Halka Dış Çap","Merkez R",
                "Strut No","Tab","Çift","Açı°","X mm","Y mm"])
    tabs = PROX_TABS if rtype=="proximal" else DIST_TABS
    for size in SIZES:
        holes = make_holes(size, angles)
        for i, h in enumerate(holes):
            tab_no = next((t["name"] for t in tabs if i in t["hole_idx"]), "")
            w.writerow([size, size+RING_WIDTH*2, h["mid_r"],
                        f"S{i+1}", tab_no, h["pair"],
                        h["angle"], h["x"], h["y"]])
    return Response("\ufeff"+out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":f"attachment;filename=hexapod_koordinatlar_{rtype}.csv"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
