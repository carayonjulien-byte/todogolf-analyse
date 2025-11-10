import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import math

app = Flask(__name__)

# --------------------------------
# CONSTANTES
# --------------------------------
# taille attendue des repères (à ajuster après impression réelle)
CALIB_MIN_AREA = 500       # trop petit -> bruit
CALIB_MAX_AREA = 20000     # trop gros -> cercle/bord
MIN_RED_AREA_FALLBACK = 100
MAX_RED_AREA_FALLBACK = 5000
NB_RINGS = 4
DEFAULT_RING_STEP_M = 5.0
# si les 3 repères sont presque sur la même ligne -> on ne leur fait pas confiance
MIN_REPERE_VERTICAL_SPREAD = 50  # px


# --------------------------------
# OUTILS
# --------------------------------
def circle_from_3_points(p1, p2, p3):
    """
    Renvoie (cx, cy, r) du cercle passant par 3 points.
    Retourne None si les points sont trop alignés.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    # IMPORTANT : bien prendre x3**2 + y3**2
    cd = (temp - (x3**2 + y3**2)) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1e-6:
        return None  # points quasi alignés

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return int(cx), int(cy), int(r)


def image_center(img):
    h, w = img.shape[:2]
    return (w // 2, h // 2)


def keep_points_in_circle(points, center, radius_px):
    if center is None or radius_px is None:
        return points
    cx, cy = center
    kept = []
    for (x, y, area) in points:
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2:
            kept.append((x, y, area))
    return kept


# --------------------------------
# DÉTECTION DES REPÈRES
# --------------------------------
def find_black_calibration_points(bgr_image):
    """
    Détecte les repères foncés (carrés/ronds) en évitant les ombres et les traits du radar.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # seuillage adaptatif = robuste aux bords sombres
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10
    )

    # petit nettoyage
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # 1) taille réaliste
        if area < CALIB_MIN_AREA or area > CALIB_MAX_AREA:
            continue

        # 2) compacité pour virer les lignes / bords
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # carré/rond ~0.7-1.0 ; trait fin << 0.5
        if circularity < 0.5:
            continue

        # 3) centre du repère
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        candidates.append((cx, cy, int(area)))

    # on garde les 3 plus gros
    candidates = sorted(candidates, key=lambda p: p[2], reverse=True)
    return candidates[:3], thresh


def order_calibration_points(calib_points):
    """1 en haut, 2 en bas."""
    if len(calib_points) < 3:
        return None
    pts = sorted(calib_points, key=lambda p: p[1])
    top = pts[0]
    bottom1, bottom2 = pts[1], pts[2]
    if bottom1[0] < bottom2[0]:
        bottom_left, bottom_right = bottom1, bottom2
    else:
        bottom_left, bottom_right = bottom2, bottom1
    return {
        "top": top,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }


def find_outer_circle_quick(bgr_image):
    """
    Fallback : on prend le plus gros contour de l'image et on l'enveloppe.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 10
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(biggest)
    return (int(x), int(y), int(radius))


def get_radar_center_and_radius_from_template_or_fallback(img, calib_struct, calib_points):
    """
    Version demandée :
    1. on ESSAIE d'abord de faire le cercle qui passe EXACTEMENT par les 3 centres détectés
    2. si ça marche pas (alignés / pas 3 points), on retombe sur ton calcul haut+bas
    3. sinon fallback sur le plus gros contour
    On retourne toujours (center, radius, used_3pt_circle: bool)
    """
    h, w = img.shape[:2]

    # 1) cercle exact à partir de 3 points
    if calib_points and len(calib_points) == 3:
        p1 = (calib_points[0][0], calib_points[0][1])
        p2 = (calib_points[1][0], calib_points[1][1])
        p3 = (calib_points[2][0], calib_points[2][1])
        circle = circle_from_3_points(p1, p2, p3)
        if circle is not None:
            cx, cy, r = circle
            return (cx, cy), r, True

    # 2) ton estimation géométrique d'avant
    if calib_struct is not None:
        top = calib_struct["top"]
        bl = calib_struct["bottom_left"]
        br = calib_struct["bottom_right"]

        bottom_y = int((bl[1] + br[1]) / 2)
        vertical_spread = bottom_y - top[1]

        if vertical_spread >= MIN_REPERE_VERTICAL_SPREAD:
            # centre = milieu horizontal des 2 du bas, et milieu vertical entre haut et bas
            center_x = int((bl[0] + br[0]) / 2)
            center_y = int((top[1] + bottom_y) / 2)
            radius = int(vertical_spread / 2)
            return (center_x, center_y), radius, False

    # 3) fallback si on n'a pas 3 repères utilisables
    oc = find_outer_circle_quick(img)
    if oc is not None:
        return (oc[0], oc[1]), oc[2], False

    # dernier recours
    return image_center(img), min(h, w) // 2, False


# --------------------------------
# POINTS ROUGES
# --------------------------------
def find_red_points(bgr_image,
                    min_area=MIN_RED_AREA_FALLBACK,
                    max_area=MAX_RED_AREA_FALLBACK):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([15, 255, 255])
    lower_red_2 = np.array([165, 70, 50])
    upper_red_2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < min_area or area_px > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ratio = w / h
        if 0.5 <= ratio <= 1.5:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            points.append((cx, cy, area_px))
    return points, mask


# --------------------------------
# RÉSUMÉ
# --------------------------------
def build_resume(coups, centre_distance=None):
    if not coups:
        return {
            "moyenne_distance_m": None,
            "moyenne_lateral_m": None,
            "dispersion_distance_m": None,
            "dispersion_lateral_m": None,
            "tendance": "aucun coup détecté"
        }

    distances = np.array([c["distance"] for c in coups], dtype=float)
    lats = np.array([c["lateral_m"] for c in coups], dtype=float)

    moyenne_distance = float(np.mean(distances))
    moyenne_lateral = float(np.mean(lats))
    dispersion_distance = float(np.std(distances, ddof=0))
    dispersion_lateral = float(np.std(lats, ddof=0))

    tendance_parts = []
    if centre_distance is not None:
        diff = moyenne_distance - centre_distance
        if diff < -1.0:
            tendance_parts.append("court")
        elif diff > 1.0:
            tendance_parts.append("long")

    if moyenne_lateral < -0.5:
        tendance_parts.append("gauche")
    elif moyenne_lateral > 0.5:
        tendance_parts.append("droite")

    tendance = "dans l'axe" if not tendance_parts else " et ".join(tendance_parts)

    return {
        "moyenne_distance_m": round(moyenne_distance, 2),
        "moyenne_lateral_m": round(moyenne_lateral, 2),
        "dispersion_distance_m": round(dispersion_distance, 2),
        "dispersion_lateral_m": round(dispersion_lateral, 2),
        "tendance": tendance
    }


# --------------------------------
# ROUTES
# --------------------------------
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API – cercle 3 points prioritaire", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    club = request.form.get("club", "Inconnu")
    centre_distance = request.form.get("centre_distance")
    centre_distance = float(centre_distance) if centre_distance else None

    # 1 m ou 5 m
    calib_mode = request.form.get("calib_mode")
    if calib_mode == "circles":
        ring_step_m = 5.0
    elif calib_mode == "squares":
        ring_step_m = 1.0
    else:
        ring_step_m = DEFAULT_RING_STEP_M

    # détecter repères
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points) if calib_points else None

    # centre + rayon (3 points d'abord)
    radar_center, outer_radius_px, used_3pt_circle = get_radar_center_and_radius_from_template_or_fallback(
        img,
        calib_struct,
        calib_points
    )

    # impacts
    shot_points, _ = find_red_points(img)
    shot_points = keep_points_in_circle(shot_points, radar_center, outer_radius_px)

    # px -> m
    max_distance_m = ring_step_m * NB_RINGS
    meters_per_px = max_distance_m / float(outer_radius_px) if outer_radius_px > 0 else 0.1

    # coups
    coups = []
    cx, cy = radar_center
    for (x, y, area) in shot_points:
        dx_px = x - cx
        lateral_m = round(dx_px * meters_per_px, 2)

        r_point_px = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        frac = r_point_px / float(outer_radius_px) if outer_radius_px > 0 else 0
        local_distance_m = round(frac * max_distance_m, 2)

        if centre_distance is not None:
            distance_finale = round(centre_distance + local_distance_m, 2)
        else:
            distance_finale = local_distance_m

        coups.append({
            "distance": distance_finale,
            "lateral_m": lateral_m
        })

    resume = build_resume(coups, centre_distance=centre_distance)

    return jsonify({
        "club": club,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "resume": resume,
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "origin_px": [float(radar_center[0]), float(radar_center[1])],
            "outer_radius_px": float(outer_radius_px),
            "used_3pt_circle": used_3pt_circle,
            "calib_points": calib_points,
        }
    })


@app.route("/mask", methods=["POST"])
def mask_route():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points) if calib_points else None
    radar_center, outer_radius_px, used_3pt_circle = get_radar_center_and_radius_from_template_or_fallback(
        img,
        calib_struct,
        calib_points
    )

    shot_points, red_mask = find_red_points(img)
    shot_points = keep_points_in_circle(shot_points, radar_center, outer_radius_px)

    overlay = img.copy()
    overlay[red_mask > 0] = (0, 0, 255)

    # cercle principal
    cv2.circle(overlay, (int(radar_center[0]), int(radar_center[1])), int(outer_radius_px), (255, 0, 255), 2)

    # cercles intermédiaires
    for i in range(1, NB_RINGS):
        r = int(outer_radius_px * (i / NB_RINGS))
        cv2.circle(overlay, (int(radar_center[0]), int(radar_center[1])), r, (0, 255, 255), 1)

    # repères
    for (x, y, area) in calib_points:
        cv2.circle(overlay, (int(x), int(y)), 6, (255, 255, 0), 2)

    # impacts
    for (x, y, area) in shot_points:
        cv2.circle(overlay, (x, y), 4, (0, 255, 0), 2)

    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmpfile.name, out)
    return send_file(tmpfile.name, mimetype="image/png")


# --------------------------------
# PAGES DE TEST
# --------------------------------
@app.route("/test")
def test_page():
    return """
    <html><body style='font-family:sans-serif'>
    <h2>Radar ToDoGolf – Test /analyze</h2>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <p><input type="file" name="image" required></p>
      <p><input type="text" name="club" placeholder="Ex: Fer7"></p>
      <p><input type="number" step="0.1" name="centre_distance" placeholder="Ex: 100"></p>
      <p>
        <select name="calib_mode">
          <option value="">(défaut 5 m)</option>
          <option value="circles">cercles (5 m)</option>
          <option value="squares">carrés (1 m)</option>
        </select>
      </p>
      <button type="submit">Analyser</button>
    </form>
    </body></html>
    """


@app.route("/test_mask")
def test_mask_page():
    return """
    <html><body style='font-family:sans-serif'>
    <h2>Radar ToDoGolf – Test /mask</h2>
    <form id="maskForm" enctype="multipart/form-data">
      <p><input type="file" id="image" name="image" required></p>
      <button type="submit">Afficher le masque</button>
    </form>
    <div style="margin-top:20px;">
      <img id="maskImage" style="max-width:100%;display:none;">
    </div>
    <script>
    const form=document.getElementById('maskForm');
    form.addEventListener('submit',async e=>{
      e.preventDefault();
      const fd=new FormData();
      fd.append('image',document.getElementById('image').files[0]);
      const res=await fetch('/mask',{method:'POST',body:fd});
      const blob=await res.blob();
      const url=URL.createObjectURL(blob);
      const img=document.getElementById('maskImage');
      img.src=url; img.style.display='block';
    });
    </script>
    </body></html>
    """


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
