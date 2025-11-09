from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile

app = FastAPI()

# ---- config radar ----
GRID_SIZE = 12          # 12 cases x 12 cases
SCALE_PER_CELL = 5.0    # 1 case = 5 m (long jeu)
FORCED_GRID_SIZE_PX = 600  # taille du carré de grille dans l'image warpée
GRID_TOP_OFFSET = 400      # position verticale de la grille dans l'image warpée


def find_markers(image_gray):
    """
    Cherche les 3 petits repères noirs imprimés sur la feuille.
    On les trie par (y,x) pour avoir: top-left, top-right, bottom-left.
    """
    _, thresh = cv2.threshold(image_gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # filtre à ajuster selon la taille imprimée de tes repères
        if 200 < area < 5000:
            markers.append((x, y, w, h, cnt))
    markers.sort(key=lambda m: (m[1], m[0]))
    return markers


def warp_sheet(image, markers):
    """
    Redresse la feuille en se basant sur les 3 repères détectés.
    On produit une image "warpée" de 1000x1400 px.
    On renvoie aussi la matrice de transfo pour pouvoir projeter les points.
    """
    (x1, y1, w1, h1, _cnt1) = markers[0]  # top-left
    (x2, y2, w2, h2, _cnt2) = markers[1]  # top-right
    (x3, y3, w3, h3, _cnt3) = markers[2]  # bottom-left

    src_pts = np.float32([
        [x1 + w1 / 2, y1 + h1 / 2],
        [x2 + w2 / 2, y2 + h2 / 2],
        [x3 + w3 / 2, y3 + h3 / 2],
        # bottom-right estimé
        [x2 + w2 / 2 + (x3 - x1), y3 + h3 / 2 + (y2 - y1)]
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
    """
    Détection du rouge sur l'image ORIGINALE (avant warp).
    On élargit la plage de rouge pour couvrir feutres / lumière chaude.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # deux plages de rouge élargies
    lower1 = np.array([0, 50, 40])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([160, 50, 40])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # nettoyage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # on prend même les petits points
        if 8 < area < 8000:
            cx = x + w // 2
            cy = y + h // 2
            pts.append((cx, cy))
    return pts


def apply_homography_to_points(points, M):
    """
    Projette les points détectés sur l'image d'origine vers l'image warpée
    en utilisant la même matrice que celle utilisée pour la feuille.
    """
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, M)
    warped_pts = warped_pts.reshape(-1, 2)
    return [(float(x), float(y)) for (x, y) in warped_pts]


def convert_points_to_grid(points, warped_shape):
    """
    Convertit les points (déjà dans l'image warpée) en coordonnées de grille.
    Ici on FORCE la position et la taille de la grille pour coller au PDF.
    """
    h, w, _ = warped_shape

    grid_size_px = FORCED_GRID_SIZE_PX
    left = (w - grid_size_px) // 2  # centré horizontalement
    top = GRID_TOP_OFFSET           # décalé vers le bas (titre au-dessus)

    cell_size = grid_size_px / GRID_SIZE

    grid_points = []
    for (px, py) in points:
        rel_x = px - left
        rel_y = py - top

        # si le point est hors du carré de grille, on l'ignore
        if rel_x < 0 or rel_y < 0 or rel_x > grid_size_px or rel_y > grid_size_px:
            continue

        cell_x = rel_x / cell_size
        cell_y = rel_y / cell_size

        # on met l'origine (0,0) au centre
        cx = cell_x - (GRID_SIZE / 2)
        cy = (GRID_SIZE / 2) - cell_y  # y vers le haut

        grid_points.append((cx, cy))

    return grid_points


@app.post("/analyse")
async def analyse_image(
    file: UploadFile = File(...),
    distance_cible: float = Form(None),
    club: str = Form(None)
):
    """
    Endpoint principal :
    - reçoit une photo (file)
    - optionnel : distance_cible (m), club (texte)
    - renvoie les coups en latéral/profondeur + distance absolue si dispo
    """
    try:
        # 1. on sauvegarde l'image reçue
        contents = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(contents)
        tmp.close()

        img = cv2.imread(tmp.name)
        if img is None:
            return JSONResponse({"error": "Impossible de lire l'image."}, status_code=400)

        # 2. on trouve les repères
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = find_markers(gray)
        if len(markers) < 3:
            return JSONResponse({"error": "Pas assez de repères détectés."}, status_code=400)

        # 3. on redresse la feuille
        warped, M = warp_sheet(img, markers)

        # 4. on détecte les points rouges sur l'image d'origine
        raw_red_points = detect_red_points_raw(img)

        # 5. on projette ces points vers l'image warpée
        warped_red_points = apply_homography_to_points(raw_red_points, M)

        # 6. on convertit en coordonnées de grille
        grid_pts = convert_points_to_grid(warped_red_points, warped.shape)

        # 7. on transforme en mètres
        coups = []
        for (gx, gy) in grid_pts:
            lateral_m = round(gx * SCALE_PER_CELL, 2)
            profondeur_m = round(gy * SCALE_PER_CELL, 2)

            # distance absolue si on a la distance cible
            distance_absolue = None
            if distance_cible is not None:
                distance_absolue = round(distance_cible + profondeur_m, 2)

            coups.append({
                "lateral_m": lateral_m,
                "profondeur_m": profondeur_m,
                "distance_absolue_m": distance_absolue
            })

        return {
            "club": club,
            "distance_cible": distance_cible,
            "nb_coups": len(coups),
            "coups": coups
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
