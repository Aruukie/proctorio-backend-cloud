"""
serverflow_koyeb.py — lightweight cloud backend for Proctorio
─────────────────────────────────────────────────────────────
Runs on Koyeb (or any cloud VPS). Handles:
- Camera registry (cameras added via frontend)
- Alert logic (receives detections from camera_agent.py)
- Appwrite logging (incidents)
- Session management
- Seat zone management

Does NOT handle:
- RTSP streams (handled by camera_agent.py on Windows laptop)
- Video recording (handled by camera_agent.py)
- Local inference (handled by inference_server_rtx.py)
"""

import os, time, threading, json, urllib.request, urllib.parse, uuid
from datetime import datetime, timezone
from fastapi import FastAPI, Query as FQuery, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Proctorio Cloud Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONF_SUSTAINED      = 0.60
CONF_IMMEDIATE      = 0.75
CONF_IMMEDIATE_BODY = 0.75
CONF_SUSTAINED_BODY = 0.60
SUSTAINED_SECONDS   = 1.5
COOLDOWN_SECONDS    = 10.0

SUSPICIOUS_CLASSES = {
    "body-shift-sus", "hand-sus", "head-lookover-forward-sus",
    "head-lookover-left-sus", "head-lookover-right-sus", "cell phone"
}
NORMAL_CLASSES  = {"body-normal", "hand-normal", "head-normal"}
PERSON_CLASSES  = {"person"}

# ─── APPWRITE ─────────────────────────────────────────────────────────────────
APPWRITE_ENDPOINT = os.environ.get("APPWRITE_ENDPOINT", "http://100.127.119.50:980/v1")
APPWRITE_PROJECT  = os.environ.get("APPWRITE_PROJECT",  "69afbc9c001f0b024b8b")
APPWRITE_DB       = os.environ.get("APPWRITE_DB",       "69afbd3a002df646f1a7")
APPWRITE_TABLE    = os.environ.get("APPWRITE_TABLE",    "incidents")
APPWRITE_KEY      = os.environ.get("APPWRITE_KEY",      "standard_941759781051805ceacd70225e00a381816a05803bbe77f710eb6b08291abfc9e685e9f9373c22aa02ff16e743af3cb6e13e979cbaeb8311fb300f148d8b47c107fb2153e6087a7250c9f2d911ae885179be72795df202c9a82e439adce1c1bbb6ac7f7d7d7172899bf010f6df91804750d82ed523efa579984aaa6a1c3c0e3e")

# ─── STATE ────────────────────────────────────────────────────────────────────
cameras: dict      = {}
cameras_lock       = threading.Lock()
session_lock       = threading.Lock()
active_session: dict = {}

# Incident queue — signals camera_agent to save clips
_incident_queue: dict = {}
_incident_queue_lock  = threading.Lock()

# Seat zones
_seat_zones: list  = []
_zones_lock        = threading.Lock()
ZONES_FILE         = "seat_zones.json"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _session_id() -> str:
    with session_lock:
        return active_session.get("session_id", "")

def _make_alert_state() -> dict:
    return {
        "lock":       threading.Lock(),
        "per_student": {},   # { student_id: { label: { start, last_log, peak_conf } } }
    }

def _make_cam_state() -> dict:
    return {
        "alert": _make_alert_state(),
        "latest_frame": b"",
    }

def _student_id(track_id: int, cam_id: str) -> str:
    """Map track_id to nearest seat zone label."""
    with _zones_lock:
        if not _seat_zones:
            return f"S{track_id}"
    return f"S{track_id}"

# ─── APPWRITE LOGGING ─────────────────────────────────────────────────────────
def log_incident(camera_id: str, incident_type: str, student_id: str,
                 confidence: float, clip_url: str = ""):
    # Push to incident queue so /detections can signal camera_agent
    with _incident_queue_lock:
        _incident_queue.setdefault(camera_id, []).append({
            "label":      incident_type,
            "session_id": _session_id(),
        })

    sess = _session_id()
    doc_id = uuid.uuid4().hex[:20]
    payload = json.dumps({
        "documentId": doc_id,
        "data": {
            "camera_id":     camera_id,
            "incident_type": incident_type,
            "session_id":    sess,
            "student_id":    student_id,
            "confidence":    round(confidence, 4),
            "status":        "UNREVIEWED",
            "clip_url":      clip_url,
        }
    }).encode()
    try:
        req = urllib.request.Request(
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DB}/collections/{APPWRITE_TABLE}/documents",
            data=payload,
            headers={
                "Content-Type":       "application/json",
                "X-Appwrite-Project": APPWRITE_PROJECT,
                "X-Appwrite-Key":     APPWRITE_KEY,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"[Appwrite] Incident logged -> {result.get('$id')} | {incident_type} | {student_id}")
    except Exception as e:
        print(f"[Appwrite] Log error: {e}")

# ─── ALERT LOGIC ──────────────────────────────────────────────────────────────
def _process_alerts(cam_id: str, dets: list, now: float, alert_state: dict):
    triggered = []
    with alert_state["lock"]:
        per_student = alert_state["per_student"]
        for d in dets:
            sid   = d.get("student_id", "S0")
            lbl   = d.get("label", "")
            conf  = d.get("conf", 0.0)
            if lbl not in SUSPICIOUS_CLASSES:
                continue

            is_body = lbl in {"hand-sus","head-lookover-forward-sus",
                               "head-lookover-left-sus","head-lookover-right-sus","body-shift-sus"}
            conf_imm  = CONF_IMMEDIATE_BODY  if is_body else CONF_IMMEDIATE
            conf_sus  = CONF_SUSTAINED_BODY  if is_body else CONF_SUSTAINED

            if conf < conf_sus:
                continue

            classes = per_student.setdefault(sid, {})
            st = classes.setdefault(lbl, {"start": None, "last_log": 0.0, "peak_conf": 0.0})
            st["peak_conf"] = max(st["peak_conf"], conf)

            # Tier A: immediate
            if conf >= conf_imm and (now - st["last_log"]) >= COOLDOWN_SECONDS:
                st["last_log"]  = now
                st["start"]     = None
                peak = st["peak_conf"]; st["peak_conf"] = 0.0
                triggered.append((sid, lbl, peak, "IMMEDIATE"))
                continue

            # Tier B: sustained
            if st["start"] is None:
                st["start"] = now
            elapsed = now - st["start"]
            if elapsed >= SUSTAINED_SECONDS and (now - st["last_log"]) >= COOLDOWN_SECONDS:
                st["last_log"]  = now
                st["start"]     = None
                peak = st["peak_conf"]; st["peak_conf"] = 0.0
                triggered.append((sid, lbl, peak, "SUSTAINED"))

    for sid, lbl, peak_conf, tier in triggered:
        print(f"[Alert:{cam_id}] {tier} | {sid} | {lbl} | conf={peak_conf:.2f}")
        threading.Thread(target=log_incident,
                         args=(cam_id, lbl, sid, peak_conf, ""),
                         daemon=True).start()

# ─── SEAT ZONES ───────────────────────────────────────────────────────────────
def _load_seat_zones():
    global _seat_zones
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE) as f:
                with _zones_lock:
                    _seat_zones = json.load(f)
            print(f"[Zones] Loaded {len(_seat_zones)} zones")
        except Exception as e:
            print(f"[Zones] Load error: {e}")

def _save_seat_zones():
    try:
        with _zones_lock:
            data = list(_seat_zones)
        with open(ZONES_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[Zones] Save error: {e}")

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    with cameras_lock:
        cam_list = [
            {"camera_id": c["camera_id"], "status": c["status"],
             "url": c.get("url", "")}
            for c in cameras.values()
        ]
    with session_lock:
        sess = dict(active_session)
    return {
        "ok": True,
        "cameras": cam_list,
        "session": sess,
        "uptime": round(time.time() - _start_time, 1),
    }

@app.get("/cameras")
def list_cameras():
    with cameras_lock:
        cam_list = [
            {"camera_id": c["camera_id"], "type": c["type"],
             "url": c.get("url",""), "ip": c.get("ip",""),
             "status": c["status"], "detecting": True, "viewing": True}
            for c in cameras.values()
        ]
    return {"cameras": cam_list, "view_camera_id": cam_list[0]["camera_id"] if cam_list else ""}

@app.get("/cameras/config")
def get_camera_config():
    """Called by camera_agent.py to get list of cameras to connect to."""
    with cameras_lock:
        cam_list = [
            {"camera_id": c["camera_id"], "url": c["url"]}
            for c in cameras.values()
            if c.get("type") == "rtsp" and c.get("url","").startswith("rtsp://")
        ]
    return {"cameras": cam_list}

class AddCameraBody(BaseModel):
    rtsp_url:  str
    camera_id: str = ""

@app.post("/cameras/add")
def add_camera(body: AddCameraBody):
    url = body.rtsp_url.strip()
    if not url.startswith("rtsp://"):
        return {"ok": False, "error": "URL must start with rtsp://"}
    try:    ip = urllib.parse.urlparse(url).hostname or "unknown"
    except: ip = "unknown"
    cid = body.camera_id.strip() or None
    with cameras_lock:
        for c in cameras.values():
            if c.get("url") == url:
                return {"ok": True, "camera_id": c["camera_id"],
                        "status": c["status"], "url": url}
        if cid is None:
            n   = sum(1 for c in cameras.values() if c["type"] == "rtsp")
            cid = f"rtsp_cam_{n + 1:02d}"
        cameras[cid] = {
            "camera_id": cid, "type": "rtsp", "url": url,
            "ip": ip, "status": "PENDING",
            "alert": _make_alert_state(),
        }
        print(f"[Camera] Registered {cid} -> {url}")
    return {"ok": True, "camera_id": cid, "status": "PENDING", "url": url,
            "message": "Camera registered — camera_agent.py will connect automatically"}

@app.post("/cameras/{camera_id}/set_view")
def set_view(camera_id: str):
    with cameras_lock:
        if camera_id not in cameras:
            return {"ok": False, "error": "Not found"}
    return {"ok": True, "view_camera_id": camera_id}

# ─── SESSION ──────────────────────────────────────────────────────────────────
class SessionBody(BaseModel):
    examiner:    str = ""
    subject:     str = ""
    room:        str = ""
    notes:       str = ""

@app.get("/session")
def get_session():
    with session_lock: return dict(active_session)

@app.post("/session")
def create_session(body: SessionBody):
    global active_session
    sid = f"SESS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
    with session_lock:
        active_session = {
            "session_id": sid,
            "examiner":   body.examiner,
            "subject":    body.subject,
            "room":       body.room,
            "notes":      body.notes,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "active":     True,
        }
    print(f"[Session] Started {sid}")
    return active_session

@app.delete("/session")
def end_session():
    with session_lock:
        active_session["active"] = False
        active_session["ended_at"] = datetime.now(timezone.utc).isoformat()
    return {"ok": True}

# ─── INCIDENTS ────────────────────────────────────────────────────────────────
@app.get("/incidents")
def get_incidents(session_id: str = FQuery(default="")):
    try:
        params = f"?queries[]=equal(\"session_id\",\"{session_id}\")" if session_id else ""
        req = urllib.request.Request(
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DB}/collections/{APPWRITE_TABLE}/documents{params}",
            headers={
                "X-Appwrite-Project": APPWRITE_PROJECT,
                "X-Appwrite-Key":     APPWRITE_KEY,
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            docs = data.get("documents", [])
            incidents = [{
                "id":            d["$id"],
                "camera_id":     d.get("camera_id",""),
                "incident_type": d.get("incident_type",""),
                "student_id":    d.get("student_id",""),
                "confidence":    d.get("confidence",0),
                "status":        d.get("status","UNREVIEWED"),
                "clip_url":      d.get("clip_url",""),
                "session_id":    d.get("session_id",""),
                "timestamp":     d.get("$createdAt",""),
            } for d in docs]
            return {"incidents": incidents}
    except Exception as e:
        print(f"[Incidents] Fetch error: {e}")
        return {"incidents": []}

@app.patch("/incidents/{incident_id}/review")
def mark_reviewed(incident_id: str):
    try:
        payload = json.dumps({"data": {"status": "REVIEWED"}}).encode()
        req = urllib.request.Request(
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DB}/collections/{APPWRITE_TABLE}/documents/{incident_id}",
            data=payload,
            headers={
                "Content-Type":       "application/json",
                "X-Appwrite-Project": APPWRITE_PROJECT,
                "X-Appwrite-Key":     APPWRITE_KEY,
            },
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/recordings")
def get_recordings(session_id: str = FQuery(default="")):
    # Recordings are clips served from camera_agent — return incident clip_urls
    try:
        params = f"?queries[]=equal(\"session_id\",\"{session_id}\")&queries[]=notEqual(\"clip_url\",\"\")" if session_id else "?queries[]=notEqual(\"clip_url\",\"\")"
        req = urllib.request.Request(
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DB}/collections/{APPWRITE_TABLE}/documents{params}",
            headers={
                "X-Appwrite-Project": APPWRITE_PROJECT,
                "X-Appwrite-Key":     APPWRITE_KEY,
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data  = json.loads(resp.read())
            docs  = data.get("documents", [])
            recs  = [{
                "id":         d["$id"],
                "session_id": d.get("session_id",""),
                "camera_id":  d.get("camera_id",""),
                "label":      d.get("incident_type",""),
                "url":        d.get("clip_url",""),
                "timestamp":  d.get("$createdAt",""),
            } for d in docs if d.get("clip_url")]
            return {"recordings": recs}
    except Exception as e:
        print(f"[Recordings] Fetch error: {e}")
        return {"recordings": []}

# ─── DETECTIONS (from camera_agent.py) ───────────────────────────────────────
class DetectionsBody(BaseModel):
    camera_id:  str
    timestamp:  str
    detections: list
    frame:      str | None = None

@app.post("/detections")
def receive_detections(body: DetectionsBody):
    cam_id = body.camera_id
    now    = time.time()

    with cameras_lock:
        if cam_id not in cameras:
            cameras[cam_id] = {
                "camera_id": cam_id, "type": "rtsp",
                "url": f"agent://{cam_id}", "ip": "agent",
                "status": "CONNECTED",
                "alert": _make_alert_state(),
            }
            print(f"[Agent] Auto-registered: {cam_id}")
        else:
            cameras[cam_id]["status"] = "CONNECTED"
        alert_state = cameras[cam_id]["alert"]

    # Parse detections
    parsed = []
    for i, p in enumerate(body.detections):
        bbox  = p.get("bbox") or [0,0,0,0]
        x,y,w,h = bbox if len(bbox)==4 else (0,0,0,0)
        label = p.get("label","")
        conf  = float(p.get("conf",0))
        sid   = f"S{i+1}"
        parsed.append({
            "x1": int(x-w/2), "y1": int(y-h/2),
            "x2": int(x+w/2), "y2": int(y+h/2),
            "label": label, "conf": conf,
            "student_id": sid,
            "suspicious": label in SUSPICIOUS_CLASSES and conf >= CONF_SUSTAINED,
        })

    _process_alerts(cam_id, parsed, now, alert_state)

    # Return incident signal if one fired
    response = {"ok": True, "camera_id": cam_id, "detections": len(parsed)}
    with _incident_queue_lock:
        if cam_id in _incident_queue and _incident_queue[cam_id]:
            inc = _incident_queue[cam_id].pop(0)
            response["incident"]       = True
            response["incident_label"] = inc["label"]
            response["session_id"]     = inc["session_id"]
    return response

# ─── SEAT ZONES ───────────────────────────────────────────────────────────────
@app.get("/seat_zones")
def get_seat_zones():
    with _zones_lock: return {"zones": list(_seat_zones)}

@app.post("/seat_zones")
def save_seat_zones(payload: dict):
    with _zones_lock:
        global _seat_zones
        _seat_zones = payload.get("zones", [])
    _save_seat_zones()
    return {"ok": True, "count": len(_seat_zones)}

@app.get("/seat_statuses")
def get_seat_statuses():
    with _zones_lock: zones = list(_seat_zones)
    statuses = {}
    for z in zones:
        statuses[z["id"]] = {
            "status":       "normal",
            "student_id":   z.get("student_num",""),
            "student_name": z.get("student_name",""),
        }
    return {"statuses": statuses}

# ─── PROXY FILE (Appwrite recordings) ────────────────────────────────────────
@app.get("/proxy/file/{file_id}")
def proxy_file(file_id: str):
    url = (f"{APPWRITE_ENDPOINT}/storage/buckets/recordings"
           f"/files/{file_id}/view?project={APPWRITE_PROJECT}")
    try:
        req = urllib.request.Request(url, headers={
            "X-Appwrite-Project": APPWRITE_PROJECT,
            "X-Appwrite-Key":     APPWRITE_KEY,
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            ct   = resp.headers.get("Content-Type","video/mp4")
        return Response(content=data, media_type=ct,
                        headers={"Cache-Control":"no-store"})
    except Exception as e:
        return Response(status_code=502, content=f"Proxy error: {e}")

# ─── STARTUP ──────────────────────────────────────────────────────────────────
_start_time = time.time()

_load_seat_zones()
print("[Server] Proctorio Cloud Backend ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
