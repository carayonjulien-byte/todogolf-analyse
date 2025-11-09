import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

CALIB_STEP_METERS = 10.0
MIN_RED_AREA = 30
MAX_RED_AREA = 4000


# ---------- détection des impacts rouges ----------
def find_red_points(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([15, 255, 255])
    lower_red_2 = np.array([165, 70, 50])
    upper_red_2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

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


# ---------- détection des 3 ronds noirs ----------
def find_black_calibration_points(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # seuil à ajuster selon ta photo
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

    # on prend les 3 plus gros
    candidates = sorted(candidates, key=lambda p: p[2], reverse=True)
    return candidates[:3], thresh


def order_calibration_points(calib_points):
    """Retourne un dict {top, bottom_left, bottom_right} ou None."""
    if len(calib_points) < 3:
        return None

    # trier par y (haut -> bas)
    pts = sorted(calib_points, key=lambda p: p[1])
    top = pts[0]
    bottom1, bottom2 = pts[1], pts[2]

    # déterminer gauche/droite
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


def compute_shot_metrics(shot_points, origin_px, meters_per_px):
    results = []
    ox, oy = origin_px
    for (x, y, area) in shot_points:
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


# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API (noir & blanc + test) OK", 200


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

    # calibration
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)

    # coups
    shot_points, _ = find_red_points(img)

    # origine + échelle
    meters_per_px = None
    origin_px = image_center(img)

    if calib_struct is not None:
        bl = calib_struct["bottom_left"]
        br = calib_struct["bottom_right"]

        # origine = milieu des 2 du bas
        origin_px = (
            int((bl[0] + br[0]) / 2),
            int((bl[1] + br[1]) / 2),
        )

        meters_per_px = compute_scale_from_bottom(bl, br)

    # fallback si pas trouvé
    if meters_per_px is None:
        meters_per_px = 0.1

    coups = compute_shot_metrics(shot_points, origin_px, meters_per_px)

    return jsonify({
        "club": club,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "meters_per_px": meters_per_px,
            "origin_px": origin_px,
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

    shot_points, red_mask = find_red_points(img)
    calib_points, _ = find_black_calibration_points(img)

    overlay = img.copy()
    # zones rouges -> rouge
    overlay[red_mask > 0] = (0, 0, 255)
    # calib -> jaune
    for (x, y, area) in calib_points:
        cv2.circle(overlay, (x, y), 10, (255, 255, 0), 2)
    # coups -> vert
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
      <body style="font-family: sans-serif;">
        <h2>Radar ToDoGolf - Test API</h2>
        <form action="/analyze" method="post" enctype="multipart/form-data">
          <p><input type="file" name="image" accept="image/*" required></p>
          <p><input type="text" name="club" placeholder="Ex: Fer7"></p>
          <button type="submit">Analyser</button>
        </form>
        <p>Tester aussi <code>/mask</code> en POST pour voir les points détectés.</p>
      </body>
    </html>
    """


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
