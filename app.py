import os
import tempfile
import math
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

# ---------------------------
# PARAMÈTRES GÉNÉRAUX
# ---------------------------
CALIB_STEP_METERS = 10.0          # distance réelle entre les deux repères du bas
CALIB_DOT_DIAMETER_MM = 3.0       # diamètre réel d’un rond noir
MIN_RED_AREA_FALLBACK = 100
MAX_RED_AREA_FALLBACK = 5000


# ---------------------------
# OUTILS GÉOMÉTRIQUES
# ---------------------------
def circle_from_3_points(p1, p2, p3):
    """Renvoie (cx, cy, r) du cercle passant par 3 points (x, y)."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]

    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1e-6:
        return None  # points quasi alignés

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return (int(cx), int(cy), int(r))


def image_center(img):
    """Retourne le centre de l’image (fallback si pas de calibration)."""
    h, w = img.shape[:2]
    return (w // 2, h // 2)


def keep_points_in_circle(points, center, radius_px):
    """Ne garde que les points dans le cercle."""
    cx, cy = center
    kept = []
    for (x, y, area) in points:
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2:
            kept.append((x, y, area))
    return kept


# ---------------------------
# OUTILS CALIBRATION
# ---------------------------
def estimate_mm_per_px_from_calib(contour_area_px, calib_diameter_mm=CALIB_DOT_DIAMETER_MM):
    if contour_area_px == 0:
        return None
    area_mm2 = np.pi * (calib_diameter_mm / 2.0) ** 2
    return np.sqrt(area_mm2 / contour_area_px)


def mm_to_px(mm_value, mm_per_px):
    if mm_per_px is None:
        return None
    return mm_value / mm_per_px


def compute_scale_from_bottom(bottom_left, bottom_right):
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    dist_px = np.sqrt(dx * dx + dy * dy)
    if dist_px == 0:
        return None
    return CALIB_STEP_METERS / dist_px


# ---------------------------
# DÉTECTION DES POINTS
# ---------------------------
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
    return {"top": top, "bottom_left": bottom_left, "bottom_right": bottom_right}


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

    # calcul aire attendue
    if mm_per_px is not None:
        min_diam_px = mm_to_px(min_diameter_mm, mm_per_px)
        max_diam_px = mm_to_px(max_diameter_mm, mm_per_px)
        min_area_px = np.pi * (min_diam_px / 2) ** 2
        max_area_px = np.pi * (max_diam_px / 2) ** 2
    else:
        min_area_px, max_area_px = MIN_RED_AREA_FALLBACK, MAX_RED_AREA_FALLBACK

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
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        points.append((cx, cy, area_px))
    return points, mask


# ---------------------------
# CALCUL DES COUPS
# ---------------------------
def compute_shot_metrics(shot_points, origin_px, meters_per_px, centre_distance=None, radar_radius_px=None):
    results = []
    ox, oy = origin_px

    for (x, y, area) in shot_points:
        dx_px = x - ox
        dy_px = oy - y  # inversion Y
        dx_m = dx_px * meters_per_px
        dy_m = dy_px * meters_per_px
        dist_m = float(math.sqrt(dx_m ** 2 + dy_m ** 2))

        angle_rad = math.atan2(dx_px, dy_px)
        angle_deg = math.degrees(angle_rad)

        dist_center = None
        if centre_distance is not None:
            diff_x = dx_m
            diff_y = dy_m - centre_distance
            dist_center = round(math.sqrt(diff_x ** 2 + diff_y ** 2), 2)

        depth_percent = None
        if radar_radius_px is not None and radar_radius_px > 0:
            depth_percent = round((dy_px / radar_radius_px) * 100.0, 1)

        results.append({
            "profondeur_m": round(dy_m, 2),
            "lateral_m": round(dx_m, 2),
            "distance_m": round(dist_m, 2),
            "distance_to_center_m": dist_center,
            "angle_deg": round(angle_deg, 1),
            "depth_percent": depth_percent,
        })

    return results


# ---------------------------
# ROUTES API
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API – calcul complet (v.11-09) OK", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    club = request.form.get("club", "Inconnu")
    centre_distance = float(request.form.get("centre_distance", 0)) or None

    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)
    mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2]) if calib_points else None
    shot_points, _ = find_red_points(img, mm_per_px=mm_per_px)

    circle_center = image_center(img)
    circle_radius = None
    if calib_struct:
        circle = circle_from_3_points(calib_struct["top"], calib_struct["bottom_left"], calib_struct["bottom_right"])
        if circle:
            circle_center = (circle[0], circle[1])
            circle_radius = circle[2]

    meters_per_px = compute_scale_from_bottom(
        calib_struct["bottom_left"], calib_struct["bottom_right"]
    ) if calib_struct else 0.1

    if circle_radius:
        shot_points = keep_points_in_circle(shot_points, circle_center, circle_radius)

    coups = compute_shot_metrics(
        shot_points, circle_center, meters_per_px,
        centre_distance=centre_distance, radar_radius_px=circle_radius
    )

    return jsonify({
        "club": club,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "nb_points_rouges": len(shot_points),
            "meters_per_px": meters_per_px,
            "origin_px": circle_center,
            "radar_radius_px": circle_radius,
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
    calib_struct = order_calibration_points(calib_points)
    mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2]) if calib_points else None
    shot_points, red_mask = find_red_points(img, mm_per_px=mm_per_px)

    circle_center = image_center(img)
    circle_radius = None
    if calib_struct:
        circle = circle_from_3_points(calib_struct["top"], calib_struct["bottom_left"], calib_struct["bottom_right"])
        if circle:
            circle_center = (circle[0], circle[1])
            circle_radius = circle[2]

    if circle_radius:
        shot_points = keep_points_in_circle(shot_points, circle_center, circle_radius)

    overlay = img.copy()
    overlay[red_mask > 0] = (0, 0, 255)

    if circle_radius:
        cv2.circle(overlay, circle_center, circle_radius, (255, 0, 255), 2)
    for (x, y, area) in calib_points:
        cv2.circle(overlay, (x, y), 5, (255, 255, 0), 2)
    for (x, y, area) in shot_points:
        cv2.circle(overlay, (x, y), 4, (0, 255, 0), 2)

    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmpfile.name, out)
    return send_file(tmpfile.name, mimetype="image/png")


# ---------------------------
# ROUTES DE TEST
# ---------------------------
@app.route("/test")
def test_page():
    return """
    <html><body style='font-family:sans-serif'>
    <h2>Radar ToDoGolf – Test /analyze</h2>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <p><input type="file" name="image" required></p>
      <p><input type="text" name="club" placeholder="Ex: Fer7"></p>
      <p><input type="number" step="0.1" name="centre_distance" placeholder="Ex: 170"></p>
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
