import cv2
import numpy as np


def _checker_score(img, ax, ay, s, i0, j0):
    """Nombre de cases (sur 64) dont la couleur de fond suit l'alternance clair/foncé."""
    ih, iw = img.shape[:2]
    cols = np.zeros((8, 8, 3))
    for r in range(8):
        for c in range(8):
            pts = []
            for dx in (-0.38, 0.38):
                for dy in (-0.38, 0.38):
                    x = int(round(ax + (i0 + c + dx) * s))
                    y = int(round(ay + (j0 + r + dy) * s))
                    if not (0 <= x < iw and 0 <= y < ih):
                        return 0
                    pts.append(img[y, x])
            cols[r, c] = np.median(pts, axis=0)
    par = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0
    c0, c1 = np.median(cols[par], axis=0), np.median(cols[~par], axis=0)
    if np.abs(c0 - c1).sum() < 40:
        return 0
    ok0 = np.abs(cols[par] - c0).sum(axis=1) < 45
    ok1 = np.abs(cols[~par] - c1).sum(axis=1) < 45
    return int(ok0.sum() + ok1.sum())


def detect_board(frame, max_width=1280, thr=10):
    """Localise un échiquier 2D (cases unies) dans une image.

    Retourne ((x0, y0, x1, y1), nb_cases_trouvées) en pixels de l'image d'origine, ou None.
    Principe : les cases sont des zones de couleur uniforme, quasi carrées, de même taille,
    alignées sur un réseau régulier de 8x8.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w)
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    ih, iw = img.shape[:2]
    img = img.astype(np.int16)
    edge = np.zeros((ih, iw), bool)
    edge[:, :-1] |= np.abs(img[:, 1:] - img[:, :-1]).max(axis=2) > thr
    edge[:-1, :] |= np.abs(img[1:] - img[:-1]).max(axis=2) > thr
    flat = (~edge).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(flat, connectivity=4)

    min_s, max_s = 10, min(ih, iw) / 8 * 1.05
    ws, hs, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    ok = (ws >= min_s) & (hs >= min_s) & (ws <= max_s) & (hs <= max_s)
    ok &= np.abs(ws - hs) <= 0.12 * np.maximum(ws, hs)
    ok &= areas >= 0.25 * ws * hs
    ok[0] = False
    ids = np.flatnonzero(ok)
    if len(ids) < 16:
        return None
    sz = (ws[ids] + hs[ids]) / 2.0
    cx = stats[ids, 0] + ws[ids] / 2.0
    cy = stats[ids, 1] + hs[ids] / 2.0

    # taille dominante des cases
    best = (0, None)
    for s0 in np.unique(np.round(sz)):
        c = int((np.abs(sz - s0) <= 0.08 * s0).sum())
        if c > best[0]:
            best = (c, s0)
    if best[0] < 16:
        return None
    s0 = best[1]
    sel = np.abs(sz - s0) <= 0.08 * s0
    cx, cy, sz = cx[sel], cy[sel], sz[sel]
    s0 = float(np.median(sz))

    # réseau : on cherche la case de référence qui aligne le plus de cases
    best = (0, 0)
    for k in range(len(cx)):
        fx = (cx - cx[k]) / s0
        fy = (cy - cy[k]) / s0
        inl = (np.abs(fx - np.round(fx)) < 0.12) & (np.abs(fy - np.round(fy)) < 0.12)
        if inl.sum() > best[0]:
            best = (int(inl.sum()), k)
    if best[0] < 16:
        return None
    k = best[1]
    fx, fy = (cx - cx[k]) / s0, (cy - cy[k]) / s0
    inl = (np.abs(fx - np.round(fx)) < 0.12) & (np.abs(fy - np.round(fy)) < 0.12)
    ii, jj = np.round(fx[inl]).astype(int), np.round(fy[inl]).astype(int)
    px, py = cx[inl], cy[inl]

    # affinage par moindres carrés : cx = ax + i*s ; cy = ay + j*s
    m = len(ii)
    A = np.zeros((2 * m, 3))
    b = np.concatenate([px, py])
    A[:m, 0], A[:m, 2] = 1, ii
    A[m:, 1], A[m:, 2] = 1, jj
    (ax, ay, s), *_ = np.linalg.lstsq(A, b, rcond=None)

    # fenêtres 8x8 candidates, départagées par l'alternance de deux couleurs
    grid = {(int(i), int(j)) for i, j in zip(ii, jj)}
    wins = []
    for i0 in range(ii.min() - 7, ii.max() + 1):
        for j0 in range(jj.min() - 7, jj.max() + 1):
            c = sum((i, j) in grid for i in range(i0, i0 + 8) for j in range(j0, j0 + 8))
            if c >= 20:
                wins.append((c, i0, j0))
    if not wins:
        return None
    top = max(c for c, _, _ in wins)
    best = None
    for c, i0, j0 in wins:
        if c < 0.7 * top:
            continue
        sc = _checker_score(img, ax, ay, s, i0, j0)
        if best is None or (sc, c) > best[:2]:
            best = (sc, c, i0, j0)
    score, count, i0, j0 = best
    if score < 20:
        return None
    x0 = ax + i0 * s - s / 2
    y0 = ay + j0 * s - s / 2
    box = np.array([x0, y0, x0 + 8 * s, y0 + 8 * s]) / scale
    if box[0] < -2 or box[1] < -2 or box[2] > w + 2 or box[3] > h + 2:
        return None
    return tuple(float(v) for v in box), count
