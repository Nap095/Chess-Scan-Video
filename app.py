"""Interface web : déposer une vidéo, récupérer le PGN de la partie."""
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

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


def _is_youtube_url(value):
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
        and not parsed.username
        and not parsed.password
    )


def _download_youtube(job_id, url, log, progress):
    try:
        import yt_dlp
    except ImportError as e:
        raise ScanError("La dépendance yt-dlp n'est pas installée.") from e

    output = UPLOAD_DIR / f"{job_id}.%(ext)s"
    log("Téléchargement de la vidéo YouTube…")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [
            lambda data: progress(float(data.get("downloaded_bytes", 0)) /
                                  max(float(data.get("total_bytes") or data.get("total_bytes_estimate") or 1), 1))
            if data.get("status") == "downloading" else None
        ],
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise ScanError(f"Téléchargement YouTube impossible : {e}") from e

    candidates = sorted(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not candidates:
        raise ScanError("YouTube n'a fourni aucun fichier vidéo exploitable.")
    return candidates[0]


def run_youtube_job(job_id, url, opts):
    path = None
    try:
        job = JOBS[job_id]
        path = _download_youtube(
            job_id,
            url,
            job["log"].append,
            lambda p: job.__setitem__("progress", round(p * 0.25, 3)),
        )
        job["log"].append("Analyse de la vidéo téléchargée…")
        job["pgn"] = scan_video(
            path,
            log=job["log"].append,
            progress=lambda p: job.__setitem__("progress", round(0.25 + p * 0.75, 3)),
            cancel=lambda: job["cancel"],
            **opts,
        )
        job["status"] = "done"
    except Cancelled:
        JOBS[job_id]["status"] = "cancelled"
    except ScanError as e:
        JOBS[job_id]["status"], JOBS[job_id]["error"] = "error", str(e)
    except Exception as e:
        JOBS[job_id]["status"], JOBS[job_id]["error"] = "error", f"Erreur interne : {e}"
    finally:
        if path:
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


@app.post("/youtube")
def youtube():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not _is_youtube_url(url):
        return jsonify(error="URL YouTube invalide ou domaine non autorisé."), 400

    opts = {
        "white": str(data.get("white", "")).strip() or "?",
        "black": str(data.get("black", "")).strip() or "?",
    }
    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {"status": "running", "progress": 0.0, "log": [], "pgn": "", "error": "", "cancel": False}
    threading.Thread(target=run_youtube_job, args=(job_id, url, opts), daemon=True).start()
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
