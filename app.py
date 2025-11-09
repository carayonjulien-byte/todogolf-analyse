from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import base64

app = Flask(__name__)

# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------

def detect_forme_reperes(contours):
    formes = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) >= 8:
            formes.append("rond")
        elif 4 <= len(approx) <= 6:
            formes.append("carre")
    if formes.count("rond") >= 3:
        return "rond"
    elif formes.count("carre") >= 3:
        return "carre"
    else:
        return "inconnu"

def calcul_centre_reperes(reperes):
    pts = np.array(reperes)
    return np.mean(pts[:, 0]), np.mean(pts[:, 1])

def detect_points_rouges(image):
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
        if r > 2:
            coups.append({"x": int(x), "y": int(y)})
    return coups

def calcul_distance_px(pt, centre):
    dx = pt["x"] - centre[0]
    dy = pt["y"] - centre[1]
    return np.sqrt(dx**2 + dy**2)

# ------------------------------------------------------------
# Routes principales
# ------------------------------------------------------------

@app.route("/")
def home():
    return "API Radar Distance ToDoGolf - OK"


# --- TEST VISUEL ---
@app.route("/test_mask", methods=["GET", "POST"])
def test_mask():
    if request.method == "GET":
        return """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <title>Test Radar Distance - Image Traitée</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f7f7f7; }
                h2 { color: #0503aa; }
                form { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
                input[type=file] { margin-bottom: 15px; }
                img { margin-top: 25px; max-width: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }
            </style>
        </head>
        <body>
            <h2>Test Radar Distance – visualisation du traitement</h2>
            <form action="/test_mask" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Analyser</button>
            </form>
        </body>
        </html>
        """

    # POST : traitement image
    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    type_repere = detect_forme_reperes(contours_
