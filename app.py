import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

# ====== PARAMS ======
MIN_RED_AREA = 30
MAX_RED_AREA = 4000
CALIB_STEP_METERS = 10.0  # adapte ça à l’écart réel entre tes 3 repères


# ====== FONCTIONS DE BASE ======
def find_red_points(bgr_image):
    """Détecte les points rouges dans l'image et renvoie une liste de (x, y, area)."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # plages de rouge un peu larges pour les photos
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([15, 255, 255])

    lower_red_2 = np.array([165, 70, 50])
    upper_red_2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # nettoyage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_RED_AREA or area > MAX_RED_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy, area))

    return points, mask


def detect_calibration_points(points, tolerance_px=25):
    """Cherche 3 points quasi alignés verticalement pour l'échelle."""
    if len(points) < 3:
        return [], points

    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                xs = [p1[0], p2[0], p3[0]]
                if max(xs) - min(xs) < tolerance_px:
                    calib = sorted([p1, p2, p3], key=lambda p: p[1])
                    calib_set = set((p[0], p[1]) for p in calib)
                    others = [p for p in points if (p[0], p[1]) not in calib_set]
                    return calib, others

    return [], points


def compute_scale_from_calib(calib_points):
    if len(calib_points) < 2:
        return None
    p_top = calib_points[0]
    p_mid = calib_points[1]
    dist_px = abs(p_mid[1] - p_top[1])
    if dist_px == 0:
        return None
    return CALIB_STEP_METERS / dist_px


def image_center(img):
    h, w = img.shape[:2]
    return (w // 2, h // 2)


def compute_shot_metrics(shots_px, origin_px, meters_per_px):
    results = []
    ox, oy = origin_px
    for (x, y, area) in shots_px:
        dx_px = x - ox
        dy_px = oy - y  # inversion Y
        dx_m = dx_px * meters_per_px
        dy_m = dy_px * meters_per_px
        dist_m = float(np.sqrt(dx_m ** 2 + dy_m ** 2))
        results.append({
            "x_m": round(dx_m, 2),
            "y_m": round(dy_m, 2),
            "distance_m": round(dist_m, 2),
            "lateral_m": round(dx_m, 2),
        })
    return results


# ====== ROUTES ======
@app.route("/", methods=["GET"])
def index():
    return "Radar Distance API (Render) OK", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """Route principale: renvoie le JSON avec les coups."""
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_storage = request.files["image"]
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    club = request.form.get("club", "Inconnu")
    distance_cible = float(request.form.get("distance_cible", 0))
    centre_distance = request.form.get("centre_distance")
    centre_distance = float(centre_distance) if centre_distance else None

    all_points, _ = find_red_points(img)
    calib_points, shot_points = detect_calibration_points(all_points)

    meters_per_px = compute_scale_from_calib(calib_points) if calib_points else None
    if meters_per_px is None:
        meters_per_px = 0.1  # fallback

    if calib_points:
        origin_px = (calib_points[0][0], calib_points[0][1])
    else:
        origin_px = image_center(img)

    coups = compute_shot_metrics(shot_points, origin_px, meters_per_px)

    return jsonify({
        "club": club,
        "distance_cible": distance_cible,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "coups": coups,
        "debug": {
            "nb_points_total": len(all_points),
            "nb_points_calib": len(calib_points),
            "meters_per_px": meters_per_px,
            "origin_px": origin_px,
        }
    })


@app.route("/mask", methods=["POST"])
def get_mask():
    """Route debug: renvoie l'image annotée avec les zones rouges trouvées."""
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_storage = request.files["image"]
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    points, mask = find_red_points(img)

    # overlay rouge
    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # dessine un rond vert sur chaque point détecté
    for (x, y, area) in points:
        cv2.circle(out, (x, y), 6, (0, 255, 0), 2)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmpfile.name, out)

    return send_file(tmpfile.name, mimetype="image/png")


if __name__ == "__main__":
    # Render donne le port dans $PORT
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
