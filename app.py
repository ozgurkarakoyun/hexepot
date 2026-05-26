import os, io, csv, math
import numpy as np
from flask import Flask, render_template, jsonify, request, Response
from correction import (
    compute_correction_program, schedule_to_csv_rows,
    calc_strut_lengths, compute_poses, resolve_cora_position,
    RING_WIDTH, HOLE_DIAM, RING_HEIGHT,
    PROX_ANGLES, DIST_ANGLES, STRUT_CONN,
    PROX_TABS, DIST_TABS, DIST_HOLE_STRUT, PAIR_LABELS
)

app   = Flask(__name__)
SIZES = [120, 150, 180, 210]

def npfix(o):
    if isinstance(o,np.integer): return int(o)
    if isinstance(o,np.floating): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,dict): return {k:npfix(v) for k,v in o.items()}
    if isinstance(o,list): return [npfix(i) for i in o]
    return o

def make_holes(diam, angles, ring_type):
    mid_r = diam/2 + RING_WIDTH/2
    strut_map = (list(range(1,7)) if ring_type=='proximal' else list(DIST_HOLE_STRUT))
    return [{
        "id":i+1,"pair":PAIR_LABELS[i],"angle":a,"strut_no":strut_map[i],
        "x":round(mid_r*math.cos(math.radians(a)),3),
        "y":round(mid_r*math.sin(math.radians(a)),3),
        "mid_r":mid_r,
    } for i,a in enumerate(angles)]

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/ring")
def api_ring():
    d=int(request.args.get("diam",150)); t=request.args.get("type","proximal")
    angles = PROX_ANGLES if t=="proximal" else DIST_ANGLES
    tabs   = PROX_TABS   if t=="proximal" else DIST_TABS
    return jsonify({
        "inner_diam":d,"outer_diam":d+RING_WIDTH*2,"ring_width":RING_WIDTH,
        "ring_height":RING_HEIGHT,"mid_r":d/2+RING_WIDTH/2,"hole_diam":HOLE_DIAM,
        "ring_type":t,"holes":make_holes(d,angles,t),"tabs":tabs,
        "strut_conn":STRUT_CONN,"prox_angles":PROX_ANGLES,"dist_angles":DIST_ANGLES,
    })

@app.route("/api/all-rings")
def api_all_rings():
    t=request.args.get("type","proximal")
    angles=PROX_ANGLES if t=="proximal" else DIST_ANGLES
    return jsonify([{"inner_diam":d,"outer_diam":d+RING_WIDTH*2,"mid_r":d/2+RING_WIDTH/2,
                     "holes":make_holes(d,angles,t)} for d in SIZES])

@app.route("/api/compute-initial",methods=["POST"])
def api_compute_initial():
    d=request.json
    try:
        (R,t),_ = compute_poses(
            float(d["separation"]), float(d["cora_dist"]),
            float(d["coronal"]),    float(d["sagittal"]),
            float(d["axial"]),      float(d["ap_trans"]),
            float(d["ml_trans"]),   float(d["length_mm"]),
            d.get("reference_ring","proximal")
        )
        return jsonify({"initial_lengths":calc_strut_lengths(int(d["ring_diam"]),R,t)})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/neutral-lengths")
def api_neutral():
    d=int(request.args.get("diam",150)); sep=float(request.args.get("sep",120))
    return jsonify({"neutral_lengths":calc_strut_lengths(d,np.eye(3),np.array([0.,0.,sep]))})

@app.route("/api/correction",methods=["POST"])
def api_correction():
    d=request.json
    try:
        r=compute_correction_program(
            ring_diam       =int(d["ring_diam"]),
            separation      =float(d["separation"]),
            cora_dist       =float(d["cora_dist"]),
            reference_ring  =d.get("reference_ring","proximal"),
            coronal         =float(d["coronal"]),
            sagittal        =float(d["sagittal"]),
            axial           =float(d["axial"]),
            ap_trans        =float(d["ap_trans"]),
            ml_trans        =float(d["ml_trans"]),
            length_mm       =float(d["length_mm"]),
            latency_days    =int(d["latency_days"]),
            correction_days =int(d["correction_days"]),
            initial_lengths =d.get("initial_lengths"),
        )
        return jsonify(npfix(r))
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/export/csv",methods=["POST"])
def export_csv():
    d=request.json
    try:
        r=compute_correction_program(
            ring_diam=int(d["ring_diam"]),separation=float(d["separation"]),
            cora_dist=float(d["cora_dist"]),reference_ring=d.get("reference_ring","proximal"),
            coronal=float(d["coronal"]),sagittal=float(d["sagittal"]),axial=float(d["axial"]),
            ap_trans=float(d["ap_trans"]),ml_trans=float(d["ml_trans"]),length_mm=float(d["length_mm"]),
            latency_days=int(d["latency_days"]),correction_days=int(d["correction_days"]),
            initial_lengths=d.get("initial_lengths"),
        )
        out=io.StringIO(); csv.writer(out).writerows(schedule_to_csv_rows(r)); out.seek(0)
        return Response("\ufeff"+out.read(),mimetype="text/csv",
            headers={"Content-Disposition":"attachment;filename=hexapod_takvim.csv"})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/export/coords")
def export_coords():
    t=request.args.get("type","proximal")
    angles=PROX_ANGLES if t=="proximal" else DIST_ANGLES
    tabs  =PROX_TABS   if t=="proximal" else DIST_TABS
    out=io.StringIO(); w=csv.writer(out)
    w.writerow(["Halka İç Çap","Halka Dış Çap","Merkez R","Strut","Tab","Açı°","X mm","Y mm"])
    for sz in SIZES:
        for i,h in enumerate(make_holes(sz,angles,t)):
            tab=next((tb["name"] for tb in tabs if i in tb["hole_idx"]),"")
            w.writerow([sz,sz+RING_WIDTH*2,h["mid_r"],f"S{h['strut_no']}",tab,h["angle"],h["x"],h["y"]])
    return Response("\ufeff"+out.getvalue(),mimetype="text/csv",
        headers={"Content-Disposition":f"attachment;filename=hexapod_{t}.csv"})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
