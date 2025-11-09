import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

# ---------------------------
# PARAMÈTRES GÉNÉRAUX
# ---------------------------
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
    if center is None or radius_px is None:
        return points
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
    """Estime mm/px à partir de l’aire d’un rond noir connu."""
    if contour_area_px == 0:
        return None
    area_mm2 = np.pi * (calib_diameter_mm / 2.0) ** 2
    return np.sqrt(area_mm2 / contour_area_px)


# ---------------------------
# DÉTECTION DES POINTS
# ---------------------------
def find_black_calibration_points(bgr_image):
    """Détecte les 3 plus gros ronds noirs du gabarit."""
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
    """Ordonne les 3 points en top / bottom_left / bottom_right."""
    if len(calib_points) < 3:
        return None
    pts = sorted(calib_points, key=lambda p: p[1])  # tri vertical
    top = pts[0]
    bottom1, bottom2 = pts[1], pts[2]
    if bottom1[0] < bottom2[0]:
        bottom_left, bottom_right = bottom1, bottom2
    else:
        bottom_left, bottom_right = bottom2, bottom1
    return {"top": top, "bottom_left": bottom_left, "bottom_right": bottom_right}


def find_red_points(bgr_image, mm_per_px=None, min_diameter_mm=2.0, max_diameter_mm=4.0):
    """Détecte les impacts rouges."""
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
        min_diam_px = min_diameter_mm / mm_per_px
        max_diam_px = max_diameter_mm / mm_per_px
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
# DÉTECTION DES 4 CERCLES DU RADAR
# ---------------------------
def find_4_concentric_circles(img, center, max_shift_px=10):
    """
    Détecte les cercles concentriques autour de `center` et renvoie
    au max 4 cercles, triés du plus petit au plus grand.
    """
    cx_ref, cy_ref = center
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=0
    )

    detected = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if abs(x - cx_ref) <= max_shift_px and abs(y - cy_ref) <= max_shift_px:
                detected.append((x, y, r))

    detected.sort(key=lambda c: c[2])
    return detected[:4]


# ---------------------------
# DISTANCE À PARTIR DES ANNEAUX
# ---------------------------
def distance_from_rings(r_point_px, ring_radii_px, ring_step_m):
    """
    Convertit un rayon en pixels -> distance en mètres à partir des cercles détectés.
    ring_radii_px doit être trié croissant.
    """
    if not ring_radii_px or ring_step_m is None:
        return None

    # on parcourt chaque cercle
    for i, r_ring in enumerate(ring_radii_px):
        if r_point_px <= r_ring:
            # point à l'intérieur de ce cercle
            if i == 0:
                # entre centre et 1er cercle → interpolation 0..1
                frac = r_point_px / r_ring if r_ring > 0 else 0
                return frac * ring_step_m
            else:
                # entre cercle i-1 et i
                r_prev = ring_radii_px[i - 1]
                span = (r_ring - r_prev) if (r_ring - r_prev) > 0 else 1
                frac = (r_point_px - r_prev) / span
                base = i * ring_step_m  # ex. i=2 → déjà 2*step
                return base - ring_step_m + frac * ring_step_m

    # si on est au delà du dernier cercle, on extrapole
    r_last = ring_radii_px[-1]
    if r_last == 0:
        return len(ring_radii_px) * ring_step_m
    frac = r_point_px / r_last
    return len(ring_radii_px) * ring_step_m * frac


# ---------------------------
# RÉSUMÉ
# ---------------------------
def build_resume(coups, centre_distance=None):
    """
    Fabrique un petit résumé à partir de la liste des coups.
    On s'attend à ce que chaque coup ait: distance, lateral_m
    """
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


# ---------------------------
# ROUTES API
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API – version cercles fixes + distance par anneaux", 200


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

    # "circles" → 5 m par cercle, "squares" → 1 m par cercle
    calib_mode = request.form.get("calib_mode")
    if calib_mode == "circles":
        ring_step_m = 5.0
    elif calib_mode == "squares":
        ring_step_m = 1.0
    else:
        ring_step_m = None  # pas précisé

    # 1. calibration points
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)

    # 2. mm/pixel (peut être None)
    mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2]) if calib_points else None

    # 3. points rouges
    shot_points, _ = find_red_points(img, mm_per_px=mm_per_px)

    # 4. centre via les 3 points noirs
    circle_center = image_center(img)
    circle_radius_from_3pts = None
    if calib_struct:
        circle = circle_from_3_points(
            calib_struct["top"],
            calib_struct["bottom_left"],
            calib_struct["bottom_right"],
        )
        if circle:
            circle_center = (circle[0], circle[1])
            circle_radius_from_3pts = circle[2]

    # 5. on cherche les 4 cercles dessinés autour de ce centre
    radar_circles = find_4_concentric_circles(img, circle_center)
    ring_radii_px = [c[2] for c in radar_circles]

    # 6. calcul de l’échelle pour le latéral
    meters_per_px = None
    scale_source = "fallback"

    if radar_circles and ring_step_m is not None:
        # on prend le plus grand cercle détecté
        max_radius_px = radar_circles[-1][2]
        ring_count = len(radar_circles)  # normalement 4
        max_radius_m = ring_step_m * ring_count
        meters_per_px = max_radius_m / float(max_radius_px)
        scale_source = "radar_4_circles"
    elif mm_per_px is not None:
        meters_per_px = mm_per_px / 1000.0
        scale_source = "black_dot_mm"
    else:
        meters_per_px = 0.1
        scale_source = "hardcoded_0.1"

    # 7. on vire les points trop dehors (si on a un rayon depuis les 3 pts)
    if circle_radius_from_3pts:
        shot_points = keep_points_in_circle(shot_points, circle_center, circle_radius_from_3pts)

    # 8. calcul des coups
    coups = []
    cx, cy = circle_center
    for (x, y, area) in shot_points:
        # latéral en m (simple)
        dx_px = x - cx
        lateral_m = round(dx_px * meters_per_px, 2)

        # rayon du point en px
        r_point_px = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # distance locale à partir des cercles (si dispo)
        if ring_radii_px and ring_step_m is not None:
            local_distance_m = distance_from_rings(r_point_px, ring_radii_px, ring_step_m)
        else:
            # fallback : on utilise la profondeur géo (oy - y)
            dy_px = cy - y
            local_distance_m = dy_px * meters_per_px

        # distance finale = distance cible + distance locale
        if centre_distance is not None and local_distance_m is not None:
            distance_finale = round(centre_distance + local_distance_m, 2)
        elif local_distance_m is not None:
            distance_finale = round(local_distance_m, 2)
        else:
            distance_finale = None

        coups.append({
            "distance": distance_finale,
            "lateral_m": lateral_m
        })

    # 9. résumé
    resume = build_resume(coups, centre_distance=centre_distance)

    return jsonify({
        "club": club,
        "centre_distance": centre_distance,
        "nb_coups": len(coups),
        "resume": resume,
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "nb_points_rouges": len(shot_points),
            "meters_per_px": meters_per_px,
            "scale_source": scale_source,
            "origin_px": list(circle_center) if circle_center else None,
            "radar_circles_px": ring_radii_px,
            "circle_radius_from_3pts": int(circle_radius_from_3pts) if circle_radius_from_3pts else None,
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

    # calib + centre
    calib_points, _ = find_black_calibration_points(img)
    calib_struct = order_calibration_points(calib_points)

    circle_center = image_center(img)
    circle_radius_from_3pts = None
    if calib_struct:
        circle = circle_from_3_points(
            calib_struct["top"],
            calib_struct["bottom_left"],
            calib_struct["bottom_right"],
        )
        if circle:
            circle_center = (circle[0], circle[1])
            circle_radius_from_3pts = circle[2]

    # mm/px
    mm_per_px = estimate_mm_per_px_from_calib(calib_points[0][2]) if calib_points else None

    # rouges
    shot_points, red_mask = find_red_points(img, mm_per_px=mm_per_px)

    # cercles dessinés
    radar_circles = find_4_concentric_circles(img, circle_center)

    # overlay
    overlay = img.copy()
    # teinte les rouges
    overlay[red_mask > 0] = (0, 0, 255)

    # cercle reconstruit
    if circle_radius_from_3pts:
        cv2.circle(overlay, circle_center, circle_radius_from_3pts, (255, 0, 255), 2)

    # cercles dessinés détectés
    for (x, y, r) in radar_circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 255), 2)

    # points de calib
    for (x, y, area) in calib_points:
        cv2.circle(overlay, (x, y), 5, (255, 255, 0), 2)

    # points rouges retenus
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
      <p><input type="number" step="0.1" name="centre_distance" placeholder="Ex: 100"></p>
      <p>
        <select name="calib_mode">
          <option value="">-- mode de calib --</option>
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
