from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------

def detect_forme_reperes(contours):
    """Détermine si les repères détectés sont ronds ou carrés."""
    formes = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) >= 8:  # forme circulaire
            formes.append("rond")
        elif 4 <= len(approx) <= 6:  # carré ou rectangle
            formes.append("carre")
    if formes.count("rond") >= 3:
        return "rond"
    elif formes.count("carre") >= 3:
        return "carre"
    else:
        return "inconnu"


def calcul_centre_reperes(reperes):
    """Retourne le centre moyen des 3 repères."""
    pts = np.array(reperes)
    return np.mean(pts[:, 0]), np.mean(pts[:, 1])


def detect_points_rouges(image):
    """Détecte les points rouges (coups) sur le radar."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coups = []
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r > 2:  # filtre petits bruits
            coups.append({"x": int(x), "y": int(y)})
    return coups


def calcul_distance_px(pt, centre):
    """Calcule la distance (en pixels) entre un point et le centre."""
    dx = pt["x"] - centre[0]
    dy = pt["y"] - centre[1]
    return np.sqrt(dx**2 + dy**2)


# ------------------------------------------------------------
# Routes principales
# ------------------------------------------------------------

@app.route("/mask", methods=["POST"])
def mask_route():
    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Détection des repères
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    type_repere = detect_forme_reperes(contours)
    print(f"[INFO] Type de repères détecté : {type_repere}")

    # Calcul du centre (simplifié)
    reperes = [cv2.minEnclosingCircle(c)[0] for c in contours[:3]]  # les 3 premiers
    centre = calcul_centre_reperes(reperes)
    print(f"[INFO] Centre estimé : {centre}")

    # Détection des coups
    coups_detectes = detect_points_rouges(img)
    print(f"[INFO] {len(coups_detectes)} coups rouges détectés")

    # --- Nouvelle logique : échelle selon forme des repères ---
    if type_repere == "rond":
        type_fiche = "radar_distance"
        echelle_m = 5
        print("[INFO] Repères ronds → Radar Distance (5m/unité)")
    elif type_repere == "carre":
        type_fiche = "radar_distance"
        echelle_m = 1
        print("[INFO] Repères carrés → Radar Distance (1m/unité)")
    else:
        type_fiche = "radar_distance"
        echelle_m = 1
        print("[WARN] Type de repères inconnu → Échelle par défaut (1m/unité)")

    # Conversion distance
    for coup in coups_detectes:
        rayon_px = calcul_distance_px(coup, centre)
        coup["rayon_px"] = rayon_px
        coup["distance_m"] = round(rayon_px * echelle_m / 100, 1)  # division pour adapter (selon résolution)

    return jsonify({
        "type_fiche": type_fiche,
        "type_repere": type_repere,
        "echelle_m": echelle_m,
        "centre": {"x": centre[0], "y": centre[1]},
        "nb_coups": len(coups_detectes),
        "coups": coups_detectes
    })


@app.route("/")
def home():
    return "API Radar Distance ToDoGolf - OK"


# ------------------------------------------------------------
# Lancement
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
