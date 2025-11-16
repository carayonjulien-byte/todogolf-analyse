import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import math

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {
        "origins": ["https://lab.todogolf.fr", "http://localhost:5500", "http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-API-Key"]
    }},
    supports_credentials=False
)

# --------------------------------
# CONSTANTES
# --------------------------------
CALIB_MIN_AREA = 150        # tolérance repères imprimés (aire min/max en px)
CALIB_MAX_AREA = 120000

# aire par défaut pour les impacts rouges (fallback global)
MIN_RED_AREA_FALLBACK = 100
MAX_RED_AREA_FALLBACK = 5000

MAX_SHOTS = 10  # nombre max de points rouges acceptés

NB_RINGS = 4  # uniquement pour le dessin du /mask, pas pour la distance réelle

# Tolérances géométrie du triangle (en pixels)
GEOM_Y_TOL = 25      # Alignement vertical des 2 points du bas
GEOM_X_TOL = 60      # Centrage horizontal du point haut vs milieu de la base
GEOM_MIN_SPREAD = 40 # Écart vertical mini (base -> top)

# Logs simples
DEBUG_LOG = True


def log_debug(msg):
    """Log minimaliste envoyé sur stdout, activable via DEBUG_LOG."""
    if DEBUG_LOG:
        print(msg, flush=True)


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
    cd = (temp - (x3**2 + y3**2)) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1e-3:
        return None

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return int(cx), int(cy), int(r)


def keep_points_in_circle(points, center, radius_px):
    if center is None or radius_px is None:
        return points
    cx, cy = center
    kept = []
    for (x, y, area) in points:
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2:
            kept.append((x, y, area))
    return kept


def compute_red_area_bounds_from_calib(calib_points):
    """
    Calcule une plage d'aire min/max pour les points rouges
    à partir de la taille des repères détectés (aires des ronds noirs/verts).

    Si pas de repère, on retombe sur le fallback global.
    """
    if not calib_points:
        return MIN_RED_AREA_FALLBACK, MAX_RED_AREA_FALLBACK

    areas = [p[2] for p in calib_points]  # p = (cx, cy, area)
    if not areas:
        return MIN_RED_AREA_FALLBACK, MAX_RED_AREA_FALLBACK

    area_ref = float(np.median(areas))
    # on autorise plus petit et plus gros, mais dans un ordre de grandeur raisonnable
    min_area_red = max(MIN_RED_AREA_FALLBACK, int(area_ref * 0.3))
    max_area_red = min(MAX_RED_AREA_FALLBACK, int(area_ref * 3.0))

    log_debug(f"[compute_red_area_bounds] area_ref={area_ref:.1f}, "
              f"min_area_red={min_area_red}, max_area_red={max_area_red}")

    return min_area_red, max_area_red


# --------------------------------
# DÉTECTION DES REPÈRES – RONDS SOMBRES UNIQUEMENT
# --------------------------------
def find_black_calibration_points(bgr_image):
    """
    Détecte les repères circulaires sombres (noir/vert foncé), en ignorant les impacts rouges.
    On ne garde que des formes suffisamment rondes.

    Retourne:
      - une liste de max 6 points: [(cx, cy, area_px), ...]
      - l'image binaire de travail (thresh), pour debug éventuel.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # léger equalize pour aider les zones sombres / claires
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < CALIB_MIN_AREA or area > CALIB_MAX_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ignorer les trucs rouges (repères = ronds sombres, potentiellement verts)
        h, s, v = hsv[cy, cx]
        is_red = ((0 <= h <= 15 and s > 60 and v > 40) or
                  (165 <= h <= 179 and s > 60 and v > 40))
        if is_red:
            continue

        per = cv2.arcLength(c, True)
        if per == 0:
            continue

        # circularité (1.0 = cercle parfait)
        circularity = 4 * np.pi * (area / (per * per))
        # ⚠ assouplie : 0.5 au lieu de 0.6
        if circularity < 0.5:
            # pas assez rond -> on rejette
            continue

        candidates.append((cx, cy, int(area)))

    # on garde les 6 plus gros ronds max,
    # la géométrie choisira ensuite le meilleur triangle
    candidates = sorted(candidates, key=lambda p: p[2], reverse=True)[:6]
    log_debug(f"[find_black_calibration_points] found={len(candidates)} candidates")

    return candidates, thresh


from itertools import combinations  # Assure-toi d'avoir cette ligne en haut du fichier

def _score_triangle_with_tolerances(top, b1, b2):
    """
    Évalue si (top, b1, b2) forme un triangle acceptable.
    Renvoie (score, reason).
    Les tolérances sont partiellement dynamiques, basées sur la longueur de la base.
    """
    # longueur horizontale de la base
    base_len = abs(b1[0] - b2[0])
    if base_len < 10:
        return None, "base-too-small"

    # alignement vertical des 2 points du bas
    dy_bottom = abs(b1[1] - b2[1])
    base_y = (b1[1] + b2[1]) / 2.0
    spread = base_y - top[1]  # distance verticale entre la base et le sommet

    x_mid_base = (b1[0] + b2[0]) / 2.0
    dx_top_mid = abs(top[0] - x_mid_base)

    # --- 🔧 TOLÉRANCES ASSOUPLIES ---
    y_tol_dyn = max(GEOM_Y_TOL, 0.25 * base_len)          # bas alignés
    x_tol_dyn = max(GEOM_X_TOL, 0.60 * base_len)          # haut centré
    min_spread_dyn = max(GEOM_MIN_SPREAD, 0.10 * base_len)  # hauteur min

    if dy_bottom > y_tol_dyn:
        return None, "bottom-misaligned"

    if spread < min_spread_dyn:
        return None, "top-not-high-enough"

    if dx_top_mid > x_tol_dyn:
        return None, "top-not-centered"

    # score = triangle bien ouvert (large + haut)
    score = base_len * spread
    return score, "ok"


def order_calibration_points(calib_points):
    """
    Cherche le meilleur triangle parmi les repères détectés.
    - On garde jusqu'à 6 repères
    - On teste toutes les combinaisons de 3
    - On choisit le triangle valide avec le meilleur score

    Retourne (structure, reason)
    """
    if not calib_points or len(calib_points) < 3:
        log_debug("[order_calibration_points] not enough calib points")
        return None, "not-enough-calib-points"

    pts = sorted(calib_points, key=lambda p: p[2], reverse=True)[:6]

    best_struct = None
    best_score = -1
    last_reason = "no-valid-triangle"

    for triplet in combinations(pts, 3):
        pA, pB, pC = triplet

        # tri par Y : top = plus haut
        pts_y = sorted(triplet, key=lambda p: p[1])
        top = pts_y[0]
        b1, b2 = pts_y[1], pts_y[2]

        score, reason = _score_triangle_with_tolerances(top, b1, b2)

        if score is None:
            last_reason = reason
            continue

        if score > best_score:
            best_score = score
            # gauche/droite
            bottom_left, bottom_right = (b1, b2) if b1[0] < b2[0] else (b2, b1)

            best_struct = {
                "top": top,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right
            }

    if best_struct is None:
        log_debug(f"[order_calibration_points] no valid triangle found, last_reason={last_reason}")
        return None, last_reason

    log_debug(f"[order_calibration_points] triangle accepted with score={best_score}")
    return best_struct, "ok"



# --------------------------------
# CENTRE/RAYON – STRICT (3 POINTS)
# --------------------------------
def get_radar_center_and_radius_strict(calib_struct, calib_points):
    """
    Version stricte : nécessite 3 repères et un triangle géométriquement valide.
    On calcule simplement le cercle passant par ces 3 points.
    """
    if calib_struct is None or not calib_points or len(calib_points) != 3:
        return None, None, False, "missing-or-invalid-points"

    p1 = (calib_struct["top"][0],           calib_struct["top"][1])
    p2 = (calib_struct["bottom_left"][0],   calib_struct["bottom_left"][1])
    p3 = (calib_struct["bottom_right"][0],  calib_struct["bottom_right"][1])
    circle = circle_from_3_points(p1, p2, p3)
    if circle is None:
        log_debug("[get_radar_center_and_radius_strict] invalid triangle (collinear points)")
        return None, None, False, "invalid-triangle"
    cx, cy, r = circle
    log_debug(f"[get_radar_center_and_radius_strict] center=({cx},{cy}), r={r}")
    return (cx, cy), r, True, "3points-ok"


# --------------------------------
# POINTS ROUGES
# --------------------------------
def find_red_points(bgr_image, min_area, max_area):
    """
    Détecte les points rouges, en filtrant par aire entre min_area et max_area.
    Retourne [(cx, cy, area_px), ...] et le mask binaire.
    """
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
        if not (0.5 <= ratio <= 1.5):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * (area_px / (peri * peri))
        if circularity < 0.5:
            continue

        cx = int(x + w / 2)
        cy = int(y + h / 2)
        points.append((cx, cy, area_px))

    log_debug(f"[find_red_points] raw_red_points={len(points)}")
    return points, mask


# --------------------------------
# ROUTES
# --------------------------------
@app.route("/", methods=["GET"])
def index():
    return "Radar ToDoGolf API – repères ronds + points rouges (distances en pixels)", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "no-image", "message": "Aucune image envoyée"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid-image", "message": "Impossible de lire l'image"}), 400

    # réduction éventuelle de l'image
    MAX_SIDE = 1600
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        log_debug(f"[analyze] image resized with scale={scale:.3f}")

    club = request.form.get("club", "Inconnu")

    # 1) repères
    calib_points, _ = find_black_calibration_points(img)
    calib_struct, calib_reason = order_calibration_points(calib_points)

    if calib_struct is None:
        # erreur géométrique explicite
        return jsonify({
            "error": "calibration-failed",
            "message": "Repères invalides : la géométrie du triangle n'est pas correcte.",
            "reason": calib_reason,
            "debug": {
                "nb_points_calib": len(calib_points) if calib_points else 0,
                "calib_points": calib_points
            }
        }), 400

    # 2) centre + rayon (STRICT, 3 points)
    radar_center, outer_radius_px, used_3pt_circle, circle_reason = get_radar_center_and_radius_strict(
        calib_struct,
        calib_points
    )
if radar_center is None or outer_radius_px is None:
    nb_calib = len(calib_points) if calib_points else 0

    # Détermination du message à donner à l'utilisateur
    if nb_calib < 3:
        # 👉 Problème typique : lumière / reflet / contraste
        tips = (
            "Moins de trois repères détectés. "
            "Essayez de reprendre la photo sans lumière directe, reflets ou flash. "
            "Utilisez une lumière diffuse."
        )
    else:
        # 👉 Trois repères détectés mais triangle KO → orientation
        tips = (
            "Les trois repères sont détectés mais leur alignement n'est pas correct. "
            "Assurez-vous que les deux repères du bas sont bien lignés en bas de la photo "
            "et que le repère du haut est au-dessus, avec la feuille à peu près droite."
        )

    return jsonify({
        "error": "calibration-failed",
        "reason": circle_reason,
        "debug": {
            "nb_points_calib": nb_calib,
            "calib_points": calib_points,
            "circle_reason": circle_reason
        },
        "tips": tips
    }), 400

    # 3) aire attendue pour les points rouges en fonction des repères
    min_red_area, max_red_area = compute_red_area_bounds_from_calib(calib_points)

    # 4) points rouges
    raw_shot_points, _ = find_red_points(img, min_red_area, max_red_area)
    # filtrage spatial : on garde uniquement ceux dans le cercle radar
    shot_points = keep_points_in_circle(raw_shot_points, radar_center, outer_radius_px)

    # 5) limite sur le nombre de points rouges
    if len(shot_points) > MAX_SHOTS:
        return jsonify({
            "error": "too-many-shots",
            "message": f"Trop de points rouges détectés à l'intérieur du radar (> {MAX_SHOTS}).",
            "reason": "too-many-red-points",
            "debug": {
                "nb_points_calib": len(calib_points),
                "calib_points": calib_points,
                "origin_px": [float(radar_center[0]), float(radar_center[1])],
                "outer_radius_px": float(outer_radius_px),
                "raw_shots_count": len(raw_shot_points),
                "shots_in_circle_count": len(shot_points),
                "min_red_area": min_red_area,
                "max_red_area": max_red_area,
            }
        }), 400

    # 6) fabrication des coups en pixels (pas de notion de mètres ici)
    cx, cy = radar_center
    coups = []
    for (x, y, area) in shot_points:
        dx_px = x - cx
        dy_px = y - cy
        dist_center_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
        r_norm = dist_center_px / float(outer_radius_px) if outer_radius_px > 0 else 0.0

        coups.append({
            "x_px": int(x),
            "y_px": int(y),
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "r_norm": float(f"{r_norm:.4f}"),
            "area_px": float(area),
        })

    return jsonify({
        "club": club,
        "nb_coups": len(coups),
        "coups": coups,
        "debug": {
            "nb_points_calib": len(calib_points),
            "calib_points": calib_points,
            "origin_px": [float(radar_center[0]), float(radar_center[1])],
            "outer_radius_px": float(outer_radius_px),
            "used_3pt_circle": used_3pt_circle,
            "circle_reason": circle_reason,
            "min_red_area": min_red_area,
            "max_red_area": max_red_area,
            "raw_shots_count": len(raw_shot_points),
        }
    })


@app.route("/mask", methods=["POST"])
def mask_route():
    if "image" not in request.files:
        return jsonify({"error": "no-image", "message": "Aucune image envoyée"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid-image", "message": "Impossible de lire l'image"}), 400

    MAX_SIDE = 1600
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        log_debug(f"[mask] image resized with scale={scale:.3f}")

    calib_points, _ = find_black_calibration_points(img)
    calib_struct, calib_reason = order_calibration_points(calib_points)

    if calib_struct is None:
        return jsonify({
            "error": "calibration-failed",
            "message": "Repères invalides pour /mask : la géométrie du triangle n'est pas correcte.",
            "reason": calib_reason,
            "debug": {
                "nb_points_calib": len(calib_points) if calib_points else 0,
                "calib_points": calib_points
            }
        }), 400

    radar_center, outer_radius_px, used_3pt_circle, circle_reason = get_radar_center_and_radius_strict(
        calib_struct,
        calib_points
    )
    if radar_center is None:
        return jsonify({
            "error": "circle-fit-failed",
            "message": "Impossible de calculer le centre du radar pour /mask.",
            "reason": circle_reason,
            "debug": {
                "nb_points_calib": len(calib_points) if calib_points else 0,
                "calib_points": calib_points
            }
        }), 400

    min_red_area, max_red_area = compute_red_area_bounds_from_calib(calib_points)
    shot_points, red_mask = find_red_points(img, min_red_area, max_red_area)
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
    <h2>Radar ToDoGolf – Test /analyze (repères ronds)</h2>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <p><input type="file" name="image" required></p>
      <p><input type="text" name="club" placeholder="Ex: Fer7"></p>
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
      if(!res.ok){
        const txt=await res.text();
        alert('Erreur: '+txt);
        return;
      }
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



