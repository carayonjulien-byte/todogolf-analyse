from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile

app = FastAPI()

GRID_SIZE = 12
GRID_MARGIN = 80
SCALE_PER_CELL = 5.0  # long jeu pour V1


def find_markers(image_gray):
    _, thresh = cv2.threshold(image_gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 200 < area < 5000:
            markers.append((x, y, w, h, cnt))
    markers.sort(key=lambda m: (m[1], m[0]))
    return markers


def warp_sheet(image, markers):
    (x1, y1, w1, h1, _cnt1) = markers[0]
    (x2, y2, w2, h2, _cnt2) = markers[1]
    (x3, y3, w3, h3, _cnt3) = markers[2]

    src_pts = np.float32([
        [x1 + w1/2, y1 + h1/2],
        [x2 + w2/2, y2 + h2/2],
        [x3 + w3/2, y3 + h3/2],
        [x2 + w2/2 + (x3 - x1), y3 + h3/2 + (y2 - y1)]
    ])

    target_w = 1000
    target_h = 1400
    dst_pts = np.float32([
        [100, 100],
        [target_w - 100, 100],
        [100, target_h - 100],
        [target_w - 100, target_h - 100]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (target_w, target_h))
    return warped, M


def detect_red_points_raw(img):
    """Détecte le rouge SUR L’IMAGE D’ORIGINE (avant warp)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # plage rouge élargie (rouge, rouge-orangé, rouge foncé)
    lower1 = np.array([0, 50, 40])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([160, 50, 40])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # on autorise même les petits points
        if 8 < area < 8000:
            cx = x + w // 2
            cy = y + h // 2
            pts.append((cx, cy))
    return pts


def apply_homography_to_points(points, M):
    """Projette les points de l'image brute vers l'image warpée."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, M)
    warped_pts = warped_pts.reshape(-1, 2)
    return [(float(x), float(y)) for (x, y) in warped_pts]


def convert_points_to_grid(points, warped_shape):
    h, w, _ = warped_shape
    grid_size_px = min(w, h) - 2 * GRID_MARGIN
    left = (w - grid_size_px) // 2
    top = (h - grid_size_px) // 2
    cell_size = grid_size_px / GRID_SIZE

    grid_points = []
    for (px, py) in points:
        rel_x = px - left
        rel_y = py - top

        cell_x = rel_x / cell_size
        cell_y = rel_y / cell_size

        cx = cell_x - (GRID_SIZE / 2)
        cy = (GRID_SIZE / 2) - cell_y
        grid_points.append((cx, cy))

    return grid_points


@app.post("/analyse")
async def analyse_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(contents)
        tmp.close()

        img = cv2.imread(tmp.name)
        if img is None:
            return JSONResponse({"error": "Impossible de lire l'image."}, status_code=400)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = find_markers(gray)
        if len(markers) < 3:
            return JSONResponse({"error": "Pas assez de repères détectés."}, status_code=400)

        # 1) on redresse
        warped, M = warp_sheet(img, markers)

        # 2) on détecte le rouge dans l'original
        raw_red_points = detect_red_points_raw(img)

        # 3) on projette ces points vers l'image redressée
        warped_red_points = apply_homography_to_points(raw_red_points, M)

        # 4) on convertit en coordonnées de grille
        grid_pts = convert_points_to_grid(warped_red_points, warped.shape)

        coups = []
        for (gx, gy) in grid_pts:
            coups.append({
                "lateral_m": round(gx * SCALE_PER_CELL, 2),
                "profondeur_m": round(gy * SCALE_PER_CELL, 2)
            })

        return {
            "nb_coups": len(coups),
            "coups": coups
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
