import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

# -----------------------
# PARAMÈTRES
# -----------------------
CALIB_STEP_METERS = 10.0
CALIB_DOT_DIAMETER_MM = 3.0  # diamètre réel du rond noir imprimé
MIN_RED_AREA_FALLBACK = 100
MAX_RED_AREA_FALLBACK = 5000
RADAR_RADIUS_FACTOR = 0.8    # % de la distance centre -> rond du haut qu'on garde


# =======================
# UTILITAIRES
# =======================
def estimate_mm_per_px_from_calib(contour_area_px, calib_diameter_mm=CALIB_DOT_DIAMETER_MM):
    if contour_area_px == 0:
        return None
    area_mm2 = np.pi * (calib_diameter_mm / 2.0) ** 2
    mm_per_px = np.sqrt(area_mm2 / contour_area_px)
    return mm_per_px


def mm_to_px(mm_value, mm_per_px):
    if mm_per_px is None:
        return None
    return mm_value / mm_per_px


def filter_points_in_circle(points, center, radius_px):
    """points = [(x,y,area)], center=(cx,cy). Garde uniquement ceux dans le disque."""
    cx, cy = center
    kept = []
    r2 = radius_px * radius_px
    for (x, y, area) in points:
        if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
            kept.append((x, y, area))
    return kept


# =======================
# DÉTECTION POINTS ROUGES
# =======================
def find_red_points(bgr_image, mm_per_px=None, min_diameter_mm=2.0, max_diameter_mm=4.0):
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

    if mm_per_px is not None:
        min_diam_px = mm_to_px(min_diameter_mm, mm_per_px)
        max_diam_px = mm_to_px(max_diameter_mm, mm_per_px)
        min_area_px = np.pi * (min_diam_px / 2) ** 2
        max_area_px = np.pi * (max_diam_px / 2) ** 2
    else:
        min_area_px = MIN_RED_AREA_FALLBACK
        max_area_px = MAX_RED_AREA_FALLBACK

    points = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < min_area_px or area_px > max_area_px:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ratio = w / h
        if ratio < 0.5 or ratio > 1.5:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy, area_px))

    return points, mask


# =======================
# DÉTECTION RONDS NOIRS
# =======================
def find_black_calibration_points(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidates.append((cx, cy, area))

    candidates = sorted(candidates, key=lambda p: p[2], reverse=True)
    return candidates[:3], thresh


def order_calibration_points(calib_points):
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


def compute_scale_from_bottom(bottom_left, bottom_right):
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    dist_px = np.sqrt(dx * dx + dy * dy)
    if dist_px == 0:
        return None
    return CALIB_STEP_METERS / dist_px


def image_center(img):
    h, w = img.shape[:2]
    return (w // 2, h // 2)


# =======================
# CALCUL COUPS
# =======================
def compute_shot_metrics(shot_points, origin_px, meters_per_px, centre_distance=None):
    results = []
    ox, oy = origin_px
    for (x, y, area) in shot_points:
        dx_px = x - ox
        dy_px = oy - y
        dx_m = dx_px * meters_per_px
        dy_m = dy_px * meters_per_px
        dist_m = float(np.sqrt(dx_m ** 2 + dy_m ** 2))

        if centre_distance is not None:
            diff_x = dx_m
            diff_y = dy_m - centre_distance
            dist_center = float(np.sqrt(diff_x ** 2 + diff_y ** 2))
            dist_center = round(dist_center, 2)
        else:
            dist_center = None

        results.append({
            "profondeur_m": round(dy_m, 2),
            "lateral_m": round(dx_m, 2),
            "distance_m": round(dist_m, 2),
            "distance_to_center_m": dist_center,
        })
    return results


# =======================
# ROUTES
# =======================
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API (filtre 2-4mm + zone) OK", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_storage = request.files["image"]
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    club = request.form.get("club", "Inconnu")
    centre_distance = request.form.get("centre_distance")
    centre_distance = float(centre_distance) if centre_distance else None

    # 1) ronds noirs
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)

    # mm/px
    mm_per_px = None
    if calib_points:
        mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2], calib_diameter_mm=CALIB_DOT_DIAMETER_MM)

    # 2) points rouges
    shot_points, _ = find_red_points(img, mm_per_px=mm_per_px, min_diameter_mm=2.0, max_diameter_mm=4.0)

    # 3) origine + rayon de zone + échelle
    meters_per_px = None
    origin_px = image_center(img)
    radar_radius_px = None

    if calib_struct is not None:
        bl = calib_struct["bottom_left"]
        br = calib_struct["bottom_right"]
        top = calib_struct["top"]

        # centre du radar
        origin_px = (
            int((bl[0] + br[0]) / 2),
            int((bl[1] + br[1]) / 2),
        )

        # rayon du radar = distance centre -> rond du haut * facteur
        dist_top = np.sqrt((top[0] - origin_px[0]) ** 2 + (top[1] - origin_px[1]) ** 2)
        radar_radius_px = int(dist_top * RADAR_RADIUS_FACTOR)

        meters_per_px = compute_scale_from_bottom(bl, br)

    if meters_per_px is None:
        meters_per_px = 0.1

    # 4) filtrer les points rouges dans le cercle
    if radar_radius_px is not None:
        shot_points = filter_points_in_circle(shot_points, origin_px, radar_radius_px)

    coups = compute_shot_metrics(shot_points, origin_px, meters_per_px, centre_distance=centre_distance)

    return jsonify({
        "club": club,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "nb_points_rouges": len(shot_points),
            "meters_per_px": meters_per_px,
            "mm_per_px": mm_per_px,
            "origin_px": origin_px,
            "radar_radius_px": radar_radius_px,
        }
    })


@app.route("/mask", methods=["POST"])
def mask_route():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_storage = request.files["image"]
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    # calib
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)

    mm_per_px = None
    origin_px = image_center(img)
    radar_radius_px = None

    if calib_points:
        mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2], calib_diameter_mm=CALIB_DOT_DIAMETER_MM)

    # rouges
    shot_points, red_mask = find_red_points(img, mm_per_px=mm_per_px, min_diameter_mm=2.0, max_diameter_mm=4.0)

    if calib_struct is not None:
        bl = calib_struct["bottom_left"]
        br = calib_struct["bottom_right"]
        top = calib_struct["top"]
        origin_px = (
            int((bl[0] + br[0]) / 2),
            int((bl[1] + br[1]) / 2),
        )
        dist_top = np.sqrt((top[0] - origin_px[0]) ** 2 + (top[1] - origin_px[1]) ** 2)
        radar_radius_px = int(dist_top * RADAR_RADIUS_FACTOR)

    # filtrer visuellement
    if radar_radius_px is not None:
        shot_points = filter_points_in_circle(shot_points, origin_px, radar_radius_px)

    overlay = img.copy()
    overlay[red_mask > 0] = (0, 0, 255)

    # cercle de zone (debug)
    if radar_radius_px is not None:
        cv2.circle(overlay, origin_px, radar_radius_px, (255, 0, 255), 2)

    # calib en jaune
    for (x, y, area) in calib_points:
        cv2.circle(overlay, (x, y), 10, (255, 255, 0), 2)

    # coups en vert
    for (x, y, area) in shot_points:
        cv2.circle(overlay, (x, y), 6, (0, 255, 0), 2)

    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmpfile.name, out)
    return send_file(tmpfile.name, mimetype="image/png")


@app.route("/test", methods=["GET"])
def test_page():
    return """
    <html>
      <body style="font-family:sans-serif;">
        <h2>Radar ToDoGolf - Test /analyze</h2>
        <form action="/analyze" method="post" enctype="multipart/form-data">
          <p><input type="file" name="image" accept="image/*" required></p>
          <p><input type="text" name="club" placeholder="Ex: Fer7"></p>
          <p><input type="number" step="0.1" name="centre_distance" placeholder="Ex: 170"></p>
          <button type="submit">Analyser</button>
        </form>
      </body>
    </html>
    """


@app.route("/test_mask", methods=["GET"])
def test_mask_page():
    return """
    <html>
      <body style="font-family:sans-serif;">
        <h2>Radar ToDoGolf - Test /mask</h2>
        <form id="maskForm" enctype="multipart/form-data">
          <p><input type="file" id="image" name="image" accept="image/*" required></p>
          <button type="submit">Afficher le masque</button>
        </form>
        <div style="margin-top:20px;">
          <img id="maskImage" style="max-width:100%; display:none;">
        </div>
        <script>
          const form = document.getElementById('maskForm');
          form.addEventListener('submit', async function(e) {
            e.preventDefault();
            const fd = new FormData();
            const fileInput = document.getElementById('image');
            fd.append('image', fileInput.files[0]);
            const res = await fetch('/mask', { method: 'POST', body: fd });
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const img = document.getElementById('maskImage');
            img.src = url;
            img.style.display = 'block';
          });
        </script>
      </body>
    </html>
    """


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
