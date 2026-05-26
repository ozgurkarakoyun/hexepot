"""
Hexapod Fiksatör — Flask Web Uygulaması
Railway deployment ready
"""

import os
import io
import csv
import json
import math
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_file

from correction import (
    compute_correction_program,
    schedule_to_csv_rows,
    calc_strut_lengths,
    RING_WIDTH, HOLE_DIAM,
    PROX_ANGLES, DIST_ANGLES, STRUT_CONN, PAIR_LABELS
)

app = Flask(__name__)

RING_SIZES = [120, 150, 180, 210]

# ─── Yardımcılar ─────────────────────────────────────────────
def numpy_to_python(obj):
    """NumPy nesnelerini JSON uyumlu Python'a çevir."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [numpy_to_python(i) for i in obj]
    return obj

def calc_holes(inner_diam, angles):
    mid_r = inner_diam / 2 + RING_WIDTH / 2
    return [{
        "id": i + 1,
        "pair": PAIR_LABELS[i],
        "angle": angle,
        "x": round(mid_r * math.cos(math.radians(angle)), 3),
        "y": round(mid_r * math.sin(math.radians(angle)), 3),
        "mid_r": mid_r,
    } for i, angle in enumerate(angles)]

# ─── Sayfalar ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── API: Halka geometrisi ────────────────────────────────────
@app.route("/api/ring", methods=["GET"])
def api_ring():
    inner_diam = int(request.args.get("diam", 150))
    ring_type  = request.args.get("type", "proximal")
    angles = PROX_ANGLES if ring_type == "proximal" else DIST_ANGLES
    holes  = calc_holes(inner_diam, angles)
    return jsonify({
        "inner_diam": inner_diam,
        "outer_diam": inner_diam + RING_WIDTH * 2,
        "ring_width": RING_WIDTH,
        "mid_r":      inner_diam / 2 + RING_WIDTH / 2,
        "hole_diam":  HOLE_DIAM,
        "ring_type":  ring_type,
        "holes":      holes,
        "pair_labels": PAIR_LABELS,
        "prox_angles": PROX_ANGLES,
        "dist_angles": DIST_ANGLES,
        "strut_conn":  STRUT_CONN,
    })

# ─── API: Tüm halkalar ────────────────────────────────────────
@app.route("/api/all-rings", methods=["GET"])
def api_all_rings():
    ring_type = request.args.get("type", "proximal")
    angles = PROX_ANGLES if ring_type == "proximal" else DIST_ANGLES
    result = []
    for d in RING_SIZES:
        result.append({
            "inner_diam": d,
            "outer_diam": d + RING_WIDTH * 2,
            "mid_r":      d / 2 + RING_WIDTH / 2,
            "holes":      calc_holes(d, angles),
        })
    return jsonify(result)

# ─── API: Düzeltme hesabı ─────────────────────────────────────
@app.route("/api/correction", methods=["POST"])
def api_correction():
    data = request.json
    try:
        result = compute_correction_program(
            ring_diam       = int(data["ring_diam"]),
            separation      = float(data["separation"]),
            cora_dist       = float(data["cora_dist"]),
            coronal         = float(data["coronal"]),
            sagittal        = float(data["sagittal"]),
            axial           = float(data["axial"]),
            ap_trans        = float(data["ap_trans"]),
            ml_trans        = float(data["ml_trans"]),
            length_mm       = float(data["length_mm"]),
            latency_days    = int(data["latency_days"]),
            correction_days = int(data["correction_days"]),
        )
        return jsonify(numpy_to_python(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── API: CSV export ─────────────────────────────────────────
@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    data = request.json
    try:
        result = compute_correction_program(**{
            k: (int(v) if k in ["ring_diam", "latency_days", "correction_days"] else float(v))
            for k, v in data.items()
        })
        rows = schedule_to_csv_rows(result)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        output.seek(0)
        return Response(
            "\ufeff" + output.read(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=hexapod_takvim.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── API: Koordinat CSV ───────────────────────────────────────
@app.route("/api/export/coords", methods=["GET"])
def export_coords():
    ring_type = request.args.get("type", "proximal")
    angles = PROX_ANGLES if ring_type == "proximal" else DIST_ANGLES
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Halka İç Çap (mm)", "Halka Dış Çap (mm)",
                     "Merkez R (mm)", "Strut", "Çift", "Açı (°)", "X (mm)", "Y (mm)"])
    for d in RING_SIZES:
        for h in calc_holes(d, angles):
            writer.writerow([d, d + RING_WIDTH*2, h["mid_r"],
                             f"S{h['id']}", h["pair"], h["angle"], h["x"], h["y"]])
    output.seek(0)
    return Response(
        "\ufeff" + output.read(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=hexapod_koordinatlar_{ring_type}.csv"}
    )

# ─── Ana giriş noktası ────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
