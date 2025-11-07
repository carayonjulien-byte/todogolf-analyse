from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile

app = FastAPI()

# config grille
GRID_SIZE = 12
GRID_MARGIN = 80
SCALE_PER_CELL = 5.0  # on force long jeu pour la V1


def find_markers(image_gray):
    # binaire inversé pour trouver les 3 repères noirs
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
    # on prend 3 repères : haut gauche, haut droite, bas gauche
    (x1, y1, w1, h1, _cnt1) = markers[0]
    (x2, y2, w2, h2, _cnt2) = markers[1]
    (x3, y3, w3, h3, _cnt3) = markers[2]

    src_pts = np.float32([
        [x1 + w1/2, y1 + h1/2],  # top-left
        [x2 + w2/2, y2 + h2/2],  # top-right
        [x3 + w3/2, y3 + h3/2],  # bottom-left
        [x2 + w2/2 + (x3 - x1), y3 + h3/2 + (y2 - y1)]  # bottom-right estimé
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
    return warped


def detect_red_points(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # nettoyage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if 20 < area < 2000:
            cx = x + w // 2
            cy = y + h // 2
            points.append((cx, cy))
    return points


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

        # on recentre (0,0) au milieu
        cx = cell_x - (GRID_SIZE / 2)
        cy = (GRID_SIZE / 2) - cell_y  # y vers le haut positif
        grid_points.append((cx, cy))

    return grid_points


@app.post("/analyse")
async def analyse_image(file: UploadFile = File(...)):
    try:
        # on sauvegarde temporairement l’image envoyée
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

        warped = warp_sheet(img, markers)

        points_px = detect_red_points(warped)
        grid_pts = convert_points_to_grid(points_px, warped.shape)

        coups = []
        for (gx, gy) in grid_pts:
            coups.append({
                "lateral_m": round(gx * SCALE_PER_CELL, 2),
                "profondeur_m": round(gy * SCALE_PER_CELL, 2)
            })

        return {
            "coups": coups,
            "nb_coups": len(coups)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
