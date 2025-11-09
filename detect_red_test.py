import cv2
import numpy as np

# >>> mets le bon chemin vers ta photo ici
IMAGE_PATH = "radar.jpg"

# paramètres à ajuster
MIN_RED_AREA = 30   # augmente à 200 si tu as du bruit
MAX_RED_AREA = 5000

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Impossible de lire l'image :", IMAGE_PATH)
        return

    # BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- 2 plages de rouge classiques ----
    lower_red_1 = np.array([0, 90, 70])
    upper_red_1 = np.array([12, 255, 255])

    lower_red_2 = np.array([165, 90, 70])
    upper_red_2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # petit nettoyage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # sauvegarde du masque pour voir si on chope le rouge
    cv2.imwrite("mask.png", mask)

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

    print(f"Points rouges détectés : {len(points)}")

    # on dessine pour vérifier
    out = img.copy()
    for (x, y, area) in points:
        cv2.circle(out, (x, y), 6, (0, 255, 0), 2)
        cv2.putText(out, f"{x},{y}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite("detected.png", out)
    print("mask.png et detected.png ont été générés.")

if __name__ == "__main__":
    main()
