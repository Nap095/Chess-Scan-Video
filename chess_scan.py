"""Extrait le PGN d'une partie d'échecs jouée en ligne à partir d'une vidéo.

Étapes :
1. localisation de l'échiquier 2D dans l'image (position et taille quelconques) ;
2. recherche de la position initiale (ignore l'introduction) et détection de
   l'orientation (blancs en haut ou en bas) ;
3. apprentissage de l'apparence des 12 pièces sur cette position initiale ;
4. suivi de la partie : à chaque image on retient le coup légal qui explique le mieux
   l'image observée ; arrêt à la fin de la partie (avant l'analyse).
"""
import argparse
import datetime
import sys
from pathlib import Path

import chess
import chess.pgn
import cv2
import numpy as np

from board_detect import detect_board

S = 32  # taille d'une case normalisée (pixels)
MIN_GAME_PLIES = 20  # une partie qui se perd en dessous est jugée non reconstituable
MIN_PLIES = 4  # en dessous, une "partie" est considérée comme un faux départ

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]
CLASS_OF = {p: i + 1 for i, p in enumerate(PIECE_ORDER)}  # 0 = case vide


class ScanError(Exception):
    pass


class Cancelled(Exception):
    pass


def _perm(flipped):
    """Pour chaque case python-chess, index de la cellule dans l'ordre d'affichage (ligne*8+colonne)."""
    out = np.empty(64, int)
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        out[sq] = (r if flipped else 7 - r) * 8 + (7 - f if flipped else f)
    return out


PERM = {False: _perm(False), True: _perm(True)}  # flipped = noirs en bas
PARITY = (np.add.outer(np.arange(8), np.arange(8)) % 2).reshape(64)  # damier en ordre d'affichage


def display_cells(frame, box):
    """Découpe l'échiquier en 64 cases (S x S), dans l'ordre d'affichage."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    img = cv2.resize(frame[y0:y1, x0:x1], (8 * S, 8 * S), interpolation=cv2.INTER_AREA).astype(np.float32)
    return img.reshape(8, S, 8, S, 3).transpose(0, 2, 1, 3, 4).reshape(64, S, S, 3)


def backgrounds(cells):
    """Couleur de fond de chaque case, estimée sur ses 4 coins."""
    k = 3
    n = cells.shape[0]
    c = np.concatenate([cells[:, :k, :k].reshape(n, -1, 3), cells[:, :k, -k:].reshape(n, -1, 3),
                        cells[:, -k:, :k].reshape(n, -1, 3), cells[:, -k:, -k:].reshape(n, -1, 3)], axis=1)
    return np.median(c, axis=1)[:, None, None, :]


def analyze_start(cells):
    """Si les cases (ordre d'affichage) montrent la position initiale, retourne
    {'flipped': noirs en bas, 'colors': couleurs de fond des 2 types de cases}, sinon None."""
    bg = backgrounds(cells)
    diff = np.abs(cells - bg).sum(axis=3)
    occ = np.minimum(diff, 120).mean(axis=(1, 2)) > 22
    expected = np.zeros(64, bool)
    expected[:16] = expected[48:] = True
    if (occ != expected).sum() > 3:
        return None

    b = bg[:, 0, 0, :]
    c = np.array([np.median(b[PARITY == p], axis=0) for p in (0, 1)])
    if np.abs(c[0] - c[1]).sum() < 40:
        return None
    if (np.abs(b - c[PARITY]).sum(axis=1) < 45).mean() < 0.85:
        return None

    # blancs/noirs : luminance des pièces des deux camps
    gray = cells @ np.array([0.114, 0.587, 0.299], np.float32)
    lum = np.full(64, np.nan)
    for i in list(range(16)) + list(range(48, 64)):
        m = diff[i] > 60
        if m.sum() >= 20:
            lum[i] = np.median(gray[i][m])
    top, bottom = np.nanmean(lum[:16]), np.nanmean(lum[48:])
    if not np.isfinite(top + bottom) or abs(top - bottom) < 40:
        return None

    mid = bg[16:48, 0, 0, :]
    colors = np.array([np.median(mid[PARITY[16:48] == p], axis=0) for p in (0, 1)])
    return {"flipped": bool(top > bottom), "colors": colors}


def board_visible(cells, colors):
    b = backgrounds(cells)[:, 0, 0, :]
    return (np.abs(b - colors[PARITY]).sum(axis=1) < 50).mean() >= 0.75


def build_templates(squares):
    """Modèles (image, masque) des 12 pièces depuis la position initiale (ordre python-chess)."""
    tmpl = {}
    bgs = backgrounds(squares)
    for sq, piece in chess.Board().piece_map().items():
        key = (piece.piece_type, piece.color)
        if key in tmpl:
            continue
        im = squares[sq]
        mask = np.abs(im - bgs[sq]).sum(axis=2) > 40
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        tmpl[key] = (im, mask)
    return tmpl


def cost_matrix(squares, tmpl):
    """D[case, classe] = écart moyen entre l'image de la case et chaque hypothèse."""
    D = np.empty((64, 13), np.float32)
    bg = backgrounds(squares)
    D[:, 0] = np.minimum(np.abs(squares - bg).sum(axis=3), 120).mean(axis=(1, 2))
    for key, (timg, mask) in tmpl.items():
        exp = np.where(mask[None, :, :, None], timg[None], bg)
        D[:, CLASS_OF[key]] = np.minimum(np.abs(squares - exp).sum(axis=3), 120).mean(axis=(1, 2))
    return D


def class_map(board):
    cm = np.zeros(64, np.int8)
    for sq, p in board.piece_map().items():
        cm[sq] = CLASS_OF[(p.piece_type, p.color)]
    return cm


START_CM = class_map(chess.Board())


def energy(D, cm):
    return float(D[np.arange(64), cm].sum())


def best_explanation(board, D, tol, max_plies=1):
    """Cherche la suite de 1 (ou 2) coups légaux dont la position finale coïncide
    avec la classification observée (écart <= tol cases) et mieux que la position
    actuelle. Retourne (nb_écarts, [coups])."""
    obs = D.argmin(axis=1)
    cur = int((class_map(board) != obs).sum())
    if cur == 0:
        return cur, []
    best = (cur, float("inf"), [])

    def consider(seq):
        nonlocal best
        cm = class_map(board)
        mism = int((cm != obs).sum())
        if mism > tol or mism >= cur:
            return
        cand = (mism, energy(D, cm), seq)
        if cand[:2] < best[:2]:
            best = cand

    for m1 in list(board.legal_moves):
        board.push(m1)
        consider([m1])
        if max_plies >= 2 and cur >= 3:
            for m2 in list(board.legal_moves):
                board.push(m2)
                consider([m1, m2])
                board.pop()
        board.pop()
    return best[0], best[2]


def candidate_boxes(cap, fps, total, progress, cancel):
    """Génère les positions d'échiquier candidates (confirmées par 2 détections concordantes)."""
    clusters = []
    for i in range(0, total, max(1, int(3 * fps))):
        if cancel():
            raise Cancelled()
        progress(i / total * 0.1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        det = detect_board(fr) or detect_board(fr, thr=20)
        if det is None:
            continue
        box = np.array(det[0])
        tol = 0.03 * (box[2] - box[0])
        for c in clusters:
            if np.abs(box - np.median(c["hits"], axis=0)).max() < tol:
                c["hits"].append(box)
                if len(c["hits"]) == 2:
                    yield tuple(np.median(c["hits"], axis=0))
                break
        else:
            clusters.append({"hits": [box]})


def find_start(cap, box, fps, begin, total, progress, cancel):
    """Première position initiale stable après `begin`. Retourne (idx, info, cells) ou None."""
    step = max(1, int(fps / 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
    prev = None
    idx = begin
    while idx < total:
        if not cap.grab():
            break
        if (idx - begin) % step == 0:
            if cancel():
                raise Cancelled()
            progress(idx / total * 0.1)
            ok, fr = cap.retrieve()
            cells = display_cells(fr, box) if ok else None
            info = analyze_start(cells) if cells is not None else None
            if info is None:
                prev = None
            elif prev is not None and np.abs(cells - prev).mean() < 2.0:
                return idx, info, cells
            else:
                prev = cells
        idx += 1
    return None


def track_game(cap, box, fps, total, start_idx, info, cells, p, log, progress, cancel):
    """Suit la partie depuis la position initiale. Retourne (coups SAN, board, idx de fin, raison de fin)."""
    flipped, colors = info["flipped"], info["colors"]
    perm = PERM[flipped]
    tmpl = build_templates(cells[perm])
    board = chess.Board()
    moves = []
    pending, pending_count, reset_count = None, 0, 0
    last_ok = start_idx / fps
    reason = "fin"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + 1)
    idx = start_idx + 1
    while idx < total and not board.is_game_over():
        if not cap.grab():
            break
        if (idx - start_idx) % p["step"] == 0:
            if cancel():
                raise Cancelled()
            if idx % (p["step"] * 25) == 0:
                progress(idx / total)
            t = idx / fps
            ok, fr = cap.retrieve()
            cells = display_cells(fr, box) if ok else None
            if cells is None or not board_visible(cells, colors):
                if t - last_ok > p["lost"]:
                    log(f"Fin de partie : échiquier disparu à {t:.1f}s")
                    break
            else:
                squares = cells[perm]
                D = cost_matrix(squares, tmpl)
                obs = D.argmin(axis=1)
                # retour à la position initiale après des coups = début de l'analyse
                if moves and int((obs != START_CM).sum()) <= p["tol"] \
                        and int((obs != class_map(board)).sum()) > p["tol"]:
                    reset_count += 1
                    if reset_count >= p["stable"]:
                        log(f"Fin de partie : échiquier réinitialisé à {t:.1f}s")
                        break
                else:
                    reset_count = 0
                mism, seq = best_explanation(board, D, p["tol"], p["plies"])
                if mism <= p["tol"] and not seq:
                    last_ok = t
                elif t - last_ok > p["lost"]:
                    log(f"Fin de partie : position inexplicable depuis {p['lost']:.0f}s ({t:.1f}s)")
                    reason = "inexplicable"
                    break
                if seq:
                    key = tuple(m.uci() for m in seq)
                    pending_count = pending_count + 1 if key == pending else 1
                    pending = key
                    if pending_count >= p["stable"]:
                        last_ok = t
                        for m in seq:
                            san = board.san(m)
                            board.push(m)
                            moves.append(san)
                            log(f"{t:8.2f}s  {len(moves):3d}. {san}")
                        pending, pending_count = None, 0
                else:
                    pending, pending_count = None, 0
        idx += 1
    return moves, board, idx, reason


def make_pgn(moves, board, white, black):
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "Partie extraite de la vidéo", "Site": "?",
        "Date": datetime.date.today().strftime("%Y.%m.%d"),
        "White": white, "Black": black,
        "Result": board.result(claim_draw=False) if board.is_game_over() else "*",
    })
    node, replay = game, chess.Board()
    for san in moves:
        node = node.add_variation(replay.parse_san(san))
        replay.push_san(san)
    return str(game) + "\n"


def scan_video(video, box=None, step=4, tol=2, stable=2, plies=1, lost=30.0,
               white="?", black="?", log=None, progress=None, cancel=None):
    """Analyse la vidéo et retourne le PGN de la partie (str).

    box : (x0, y0, x1, y1) pour imposer la position de l'échiquier (sinon détection automatique).
    log(str) : messages ; progress(float 0..1) ; cancel() -> bool pour interrompre.
    """
    log = log or (lambda s: None)
    progress = progress or (lambda x: None)
    cancel = cancel or (lambda: False)
    params = {"step": step, "tol": tol, "stable": stable, "plies": plies, "lost": lost}

    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ScanError("Vidéo illisible.")

        boxes = iter([tuple(box)]) if box else candidate_boxes(cap, fps, total, progress, cancel)
        best = None  # (moves, board)
        found_board = found_start = False
        for b in boxes:
            found_board = True
            log(f"Échiquier détecté : x={b[0]:.0f} y={b[1]:.0f} taille={b[2] - b[0]:.0f}px")
            begin = 0
            while True:
                start = find_start(cap, b, fps, begin, total, progress, cancel)
                if start is None:
                    break
                found_start = True
                idx, info, cells = start
                log(f"Position initiale à {idx / fps:.1f}s : "
                    f"{'noirs' if info['flipped'] else 'blancs'} en bas")
                moves, bd, end, reason = track_game(cap, b, fps, total, idx, info, cells, params, log, progress, cancel)
                if best is None or len(moves) > len(best[0]):
                    best = (moves, bd, reason)
                if len(moves) >= MIN_PLIES:
                    break
                log("Faux départ (trop peu de coups), recherche d'une autre position initiale…")
                begin = max(end, idx + int(fps))
            if best and len(best[0]) >= MIN_PLIES:
                break
    finally:
        cap.release()

    if not found_board:
        raise ScanError("Aucun échiquier détecté dans la vidéo.")
    if not found_start:
        raise ScanError("Échiquier trouvé, mais jamais en position initiale : "
                        "le début de la partie n'est pas visible dans la vidéo.")
    if not best or not best[0]:
        raise ScanError("Aucun coup détecté.")
    if len(best[0]) < MIN_PLIES or (best[2] == "inexplicable" and len(best[0]) < MIN_GAME_PLIES):
        raise ScanError(
            f"Impossible de reconstituer la partie : seuls {len(best[0])} demi-coups ont pu être lus "
            "avant que la position de l'échiquier ne devienne incohérente (variantes jouées, "
            "retours en arrière ou annotations sur l'échiquier ?).")
    progress(1.0)
    return make_pgn(best[0], best[1], white, black)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="fichier vidéo (défaut : 1er .mp4 de ./video)")
    ap.add_argument("-o", "--output", help="fichier PGN de sortie")
    ap.add_argument("--step", type=int, default=4, help="analyser 1 image sur N (défaut 4)")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="imposer la position de l'échiquier (sinon détection automatique)")
    ap.add_argument("--tol", type=int, default=2, help="nb de cases mal classées toléré (flèches, curseur)")
    ap.add_argument("--stable", type=int, default=2, help="nb d'analyses consécutives pour valider un coup")
    ap.add_argument("--plies", type=int, choices=(1, 2), default=1,
                    help="2 = tolère un coup manqué entre deux analyses (plus lent)")
    ap.add_argument("--lost", type=float, default=30.0,
                    help="secondes sans position cohérente (ou sans échiquier) avant de conclure à la fin de la partie")
    ap.add_argument("--white", default="?")
    ap.add_argument("--black", default="?")
    args = ap.parse_args()

    here = Path(__file__).parent
    video = Path(args.video) if args.video else next((here / "video").glob("*.mp4"))
    out = Path(args.output) if args.output else here / "output" / (video.stem + ".pgn")
    out.parent.mkdir(exist_ok=True)
    try:
        pgn = scan_video(video, box=args.box, step=args.step, tol=args.tol, stable=args.stable,
                         plies=args.plies, lost=args.lost, white=args.white, black=args.black,
                         log=lambda s: print(s, file=sys.stderr))
    except ScanError as e:
        sys.exit(str(e))
    out.write_text(pgn, encoding="utf-8")
    print(f"PGN -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
