"""Interface web : déposer une vidéo, récupérer le PGN de la partie."""
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from chess_scan import Cancelled, ScanError, scan_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3  # 4 Go
UPLOAD_DIR = Path(tempfile.gettempdir()) / "chess_scan_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

JOBS = {}
LOCK = threading.Lock()


def run_job(job_id, path, opts):
    job = JOBS[job_id]

    def log(msg):
        job["log"].append(msg)

    def progress(p):
        job["progress"] = round(p, 3)

    try:
        job["pgn"] = scan_video(path, log=log, progress=progress, cancel=lambda: job["cancel"], **opts)
        job["status"] = "done"
    except Cancelled:
        job["status"] = "cancelled"
    except ScanError as e:
        job["status"], job["error"] = "error", str(e)
    except Exception as e:  # erreur inattendue : on la remonte à l'interface
        job["status"], job["error"] = "error", f"Erreur interne : {e}"
    finally:
        path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="Aucun fichier reçu."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Format non supporté ({ext or 'inconnu'})."), 400

    opts = {
        "white": request.form.get("white", "").strip() or "?",
        "black": request.form.get("black", "").strip() or "?",
    }
    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}{ext}"
    f.save(path)
    with LOCK:
        JOBS[job_id] = {"status": "running", "progress": 0.0, "log": [], "pgn": "", "error": "", "cancel": False}
    threading.Thread(target=run_job, args=(job_id, path, opts), daemon=True).start()
    return jsonify(job=job_id)


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Tâche inconnue."), 404
    since = request.args.get("since", 0, type=int)
    return jsonify(status=job["status"], progress=job["progress"], pgn=job["pgn"], error=job["error"],
                   log=job["log"][since:], next=len(job["log"]))


@app.post("/cancel/<job_id>")
def cancel(job_id):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
